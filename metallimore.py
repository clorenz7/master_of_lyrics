import argparse
import glob
import os
import random
import shutil

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import functional as F

from model import TransCORmer
import datasets

CHAR_TOKENIZER_FILE = 'char_tokenizer.joblib'
SHAKES_TOKENIZER_FILE = 'shakes_char_tokenizer.joblib'


# TODO: Use tiktoken to train a SimpleBytePairEncoding on the data corpus


@torch.no_grad()
def evaluate(model, data_loader, loss_obj, device='cpu'):
    model.eval()
    losses = []
    # for song_tokens in dataset:
    for song_tokens in data_loader:
        song_tokens = song_tokens.to(device)

        output = model(song_tokens)
        labels = song_tokens[:, 1:]

        # Don't use the last output since there is no label
        aligned_output = output[:, :-1, :]
        # Transpose last channel and seq dimension to work with loss
        aligned_output = aligned_output.transpose(-2, -1)

        loss = loss_obj(aligned_output, labels)
        losses.append(loss.item())

    model.train()

    return np.mean(losses)


def train(model, dataset, train_params, device='cpu', val_dataset=None,
          batch_size=16):

    optimizer = AdamW(
        model.parameters(),
        lr=train_params.get('lr', 1e-3),
    )

    train_loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=0,
        generator=torch.Generator(),  # For 1-1 reproduction
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=0,
        generator=torch.Generator(),
    )

    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    model.to(device)

    loss_obj = nn.CrossEntropyLoss()

    epoch_losses = []
    n_epochs = train_params.get('n_epochs', 10)

    tr_loss = evaluate(
        model, train_loader, loss_obj, device=device
    )
    val_loss = evaluate(
        model, val_loader, loss_obj, device=device
    )
    print_str = f'Epoch #{0} done! Avg Train Loss: {tr_loss:0.4f}'
    print_str += f' Val loss: {val_loss:0.4f}'
    print(print_str)

    for e_idx in range(n_epochs):
        losses = []
        for b_idx, song_tokens in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)
            song_tokens = song_tokens.to(device)

            output = model(song_tokens)
            labels = song_tokens[:, 1:]

            # Don't use the last output since there is no implicit label
            # TODO: Pass the label for each token through the data loader
            aligned_output = output[:, :-1, :]
            # Transpose last channel and seq dimension to work with loss
            aligned_output = aligned_output.transpose(-2, -1)

            loss = loss_obj(aligned_output, labels)
            losses.append(loss.item())

            loss.backward()
            optimizer.step()

        avg_loss = np.mean(losses)
        epoch_losses.append(avg_loss)
        tr_loss = evaluate(model, train_loader, loss_obj, device=device)
        print_str = f'Epoch #{e_idx+1} done! Avg Train Loss: {tr_loss:0.4f}'

        if val_dataset is not None:
            val_loss = evaluate(model, val_loader, loss_obj, device=device)
            print_str += f' Val loss: {val_loss:0.4f}'

        print(print_str)


@torch.no_grad()
def generate_lyrics(model, title, tokenizer, max_tokens=2000, device='cpu',
                    temp=1.0, top_k=None, window_size=32, shakes_mode=True,
                    eos_token=None):

    model.eval()

    if title is None:
        title = tokenizer.decode([0])
        lyrics = title
    else:
        if shakes_mode:
            lyrics = f'{title.upper()}:\n'
        else:
            lyrics = f'## "{title.upper()}"\n'

    print(lyrics, end="", flush=True)
    char = ""

    eos_token = eos_token or tokenizer.eos_token

    tokens = tokenizer.encode(lyrics)
    tokens = torch.tensor(tokens).unsqueeze_(0)
    tokens = tokens.to(device)

    while char != eos_token and len(lyrics) < max_tokens:
        output = model(tokens[:, -(window_size-1):])
        logits = output[:, -1, :] / temp
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        token_idx = torch.multinomial(probs, num_samples=1)
        tokens = torch.hstack((tokens, token_idx))
        char = tokenizer.decode([token_idx.detach().item()])
        lyrics += char
        print(char, end="", flush=True)

    return lyrics


def main():
    parser = argparse.ArgumentParser(
        'Script for creating a Metallica Lyric generator'
    )
    parser.add_argument(
        'data_dir', metavar='data_dir',
        help='Directory containing files with .txt lyrics files'
    )
    parser.add_argument('--make_char_tokenizer', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--replicate', action='store_true',
                        help="Use this to replicate Shakespeare results")
    parser.add_argument(
        '--split', type=str,
        help="Split lyrics into 80/20 train/test and copy to this directory"
    )
    parser.add_argument(
        '--pretrain', type=str,
        help="Data file to pretrain the model. If ends with .pth, will load it"
    )
    parser.add_argument(
        '--save', type=str, default='metallimore',
        help='Location where model is stored. ".pth" will be added to end. '
             'Will be saved in train mode, will be loaded for eval mode.'
    )
    parser.add_argument(
        '-d', '--dropout', type=float, default=0.0,
        help='Amount of dropout to apply'
    )
    parser.add_argument(
        '--lr', type=float, default=2e-5,
        help='Learning rate to finetune with'
    )
    parser.add_argument(
        '-e', '--n_epochs', type=int, default=20,
        help='Number of epochs to finetune with'
    )
    parser.add_argument(
        '--eval', type=str, default=None,
        help='comma separated list of song titles to generate from'
    )

    torch.manual_seed(1337)

    cli_args = parser.parse_args()
    if cli_args.make_char_tokenizer:
        print('Making the tokenizer!')
        if cli_args.replicate:
            willy_tokens = datasets.get_shakes_tokens(do_clean=False)
            token_map, all_chars = datasets.create_char_tokenizer(
                None,
                pretrain_tokens=willy_tokens
            )
            joblib.dump([token_map, all_chars], SHAKES_TOKENIZER_FILE)
        else:
            willy_tokens = datasets.get_shakes_tokens()
            if cli_args.split:
                data_dir = cli_args.data_dir
            else:
                data_dir = os.path.join(cli_args.data_dir, '**')
            token_map, all_chars = datasets.create_char_tokenizer(
                data_dir,
                # '~\Dropbox\data\Metallica_Lyrics',
                # os.path.join(cli_args.data_dir, '**'),
                pretrain_tokens=willy_tokens
            )
            joblib.dump([token_map, all_chars], CHAR_TOKENIZER_FILE)

    # Split the data
    if cli_args.split:
        train_dir = os.path.join(cli_args.split, "train")
        test_dir = os.path.join(cli_args.split, "test")
        os.makedirs(train_dir)
        os.makedirs(test_dir)
        file_list = glob.glob(os.path.join(cli_args.data_dir, "*.txt"))

        random.shuffle(file_list)
        n_test = int(0.2 * len(file_list))
        for file_name in file_list[:n_test]:
            new_name = os.path.join(test_dir, os.path.basename(file_name))
            shutil.copy(file_name, new_name)

        for file_name in file_list[n_test:]:
            new_name = os.path.join(train_dir, os.path.basename(file_name))
            shutil.copy(file_name, new_name)
    else:
        train_dir = os.path.join(cli_args.data_dir, "train")
        test_dir = os.path.join(cli_args.data_dir, "test")

    if cli_args.replicate:
        tokenizer = datasets.CharTokenizer(SHAKES_TOKENIZER_FILE)
    else:
        tokenizer = datasets.CharTokenizer(CHAR_TOKENIZER_FILE)

    n_tokens = len(tokenizer)
    # Andrej's final
    n_positions = 256
    n_embed = 384
    n_layers = 6
    n_heads = 6
    batch_size = 64
    dropout = 0.2
    lr = 3e-4
    n_epochs = 30

    # Andrej's initial settings:
    # Eval every 100
    # iters = 5000
    # learn rate = 1e-3
    # eval_iters = 200
    # # For replication:
    # n_positions = 32
    # n_embed = 64
    # n_heads = 4
    # n_layers = 4
    # batch_size = 16
    # dropout = 0.0
    # lr = 1e-3
    # n_epochs = 50

    dropout = cli_args.dropout

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = TransCORmer(
        n_tokens, n_embed=n_embed, n_positions=n_positions,
        n_blocks=n_layers, dropout=dropout, n_heads=n_heads
    )

    if cli_args.pretrain:
        if cli_args.pretrain.endswith('.pth'):
            print("Loading pretrained model")
            prev_state = torch.load(cli_args.pretrain).state_dict()
            model.load_state_dict(prev_state)
            model.to(device)

        else:
            print("Running pre-training!")
            with open(cli_args.pretrain, 'r') as fp:
                text = fp.read()
                if not cli_args.replicate:
                    text = datasets.clean_willy(text)
                tokens = tokenizer.encode(text)
            split_idx = int(len(tokens) * 0.9)

            train_dataset = datasets.WillyShakesDataset(
                tokens[:split_idx], window_size=n_positions,
                size=100*batch_size
            )
            test_dataset = datasets.WillyShakesDataset(
                tokens[split_idx:], window_size=n_positions,
                deterministic=888, size=200*batch_size
            )
            train_params = {'n_epochs': n_epochs, 'lr': lr}
            train(
                model, train_dataset, train_params,
                device=device, val_dataset=test_dataset,
                batch_size=batch_size
            )
            file_name = f'{cli_args.save}.pretrained.pth'
            torch.save(model, file_name)

            print("Example Shakespeare:")
            lyrics = generate_lyrics(
                model, None, tokenizer, device=device,
                shakes_mode=True, eos_token='~'
            )

    train_dataset = datasets.MetallicaLyricsDataset(
        train_dir, tokenizer, cat_mode=False, window_size=n_positions,
        size=10*batch_size
    )
    test_dataset = datasets.MetallicaLyricsDataset(
        test_dir, tokenizer, cat_mode=False, window_size=n_positions,
        size=10*batch_size
    )

    file_name = f'{cli_args.save}.pth'
    if cli_args.train:
        print('Training Metallimore!')
        train_params = {
            'n_epochs': cli_args.n_epochs,
            'lr': cli_args.lr
        }
        try:
            train(
                model, train_dataset, train_params,
                device=device, val_dataset=test_dataset,
                batch_size=batch_size
            )
        except KeyboardInterrupt:
            # So that you can break out if model starting to overtrain
            pass

        torch.save(model, file_name)
    else:
        model = torch.load(file_name)

    titles = None if not(cli_args.eval) else cli_args.eval.split(",")

    titles = titles or [
        "the forgotten legends",
        "Coroner",
        "nothing else unforgiven",
        "master of lyrics",
        "enter sadman",
        "two",
        "transformer",
        "unforgiven x",
        "too fast too furious",
        "wrecking ball",
        "firework",
        "bad blood",
        "poker face",
    ]

    for title in titles:
        lyrics = generate_lyrics(
            model, title=title, tokenizer=tokenizer,
            device=device, temp=0.8, top_k=20,
            window_size=n_positions, shakes_mode=True,
            max_tokens=1000
        )
        print("\n-------------------------------------\n")


if __name__ == '__main__':
    main()
