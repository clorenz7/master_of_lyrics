import argparse
import glob
import math
import os
import random
import re
import shutil

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from model import TransCORmer

CHAR_TOKENIZER_FILE = 'char_tokenizer.joblib'
SHAKES_TOKENIZER_FILE = 'shakes_char_tokenizer.joblib'


# TODO: Use tiktoken to train a SimpleBytePairEncoding on the data corpus

def remove_bracketed_text(text):
    """
    Removes comments like [solo] or [Chorus] from song lyrics
    """
    pattern = r"\s*\[.*?\]\s*"
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text

def clean_lyrics(all_lyrics):
    """
    Replace rare and weird characters.
    Should get down to 70 character vocabulary.
    """
    all_lyrics = remove_bracketed_text(all_lyrics)

    # Remove rare chars and replace unicodes
    all_lyrics = all_lyrics.replace('‘', "'")
    all_lyrics = all_lyrics.replace("&", "AND")
    all_lyrics = all_lyrics.replace('“', '"')
    all_lyrics = all_lyrics.replace('”', '"')
    all_lyrics = all_lyrics.replace('…', "...")
    all_lyrics = all_lyrics.replace('...', "")
    all_lyrics = all_lyrics.replace('—', '-')
    all_lyrics = all_lyrics.replace('–', '-')
    all_lyrics = all_lyrics.replace('/', '-')
    all_lyrics = all_lyrics.replace('-', '-')
    all_lyrics = all_lyrics.replace('é', 'e')
    all_lyrics = all_lyrics.replace('Æ', 'Ae')
    all_lyrics = all_lyrics.replace('\n ', '\n')
    all_lyrics = all_lyrics.replace('\n\n', '\n')
    all_lyrics = all_lyrics.replace('  ', ' ')
    all_lyrics = all_lyrics.replace('(', '')
    all_lyrics = all_lyrics.replace(')', '')

    return all_lyrics


class CharTokenizer(object):

    def __init__(self, token_map_file, eos_token=';'):
        self.encode_map, self.decode_list = joblib.load(token_map_file)
        self.eos_token = eos_token

    def encode(self, text):
        return [self.encode_map[c] for c in text]

    def decode(self, tokens):
        return "".join(self.decode_list[t] for t in tokens)

    def __len__(self):
        return len(self.decode_list)


class MetallicaLyricsDataset(Dataset):

    _tokenizer_class_map = {
        'char_tokenizer': CharTokenizer
    }

    def __init__(self, data_dir, tokenizer=None):
        file_list = glob.glob(os.path.join(os.path.expanduser(data_dir), '*.txt'))

        self.all_text = []
        self.all_tokens = []

        self.tokenizer = tokenizer

        for file_name in file_list:
            with open(file_name, 'r') as fp:
                song_text = clean_lyrics(fp.read())
                self.all_text.append(song_text)

            if tokenizer is not None:
                self.all_tokens.append(
                    tokenizer.encode(song_text + tokenizer.eos_token)
                )

    def get_all(self):
        """
        Returns the lyrics from all songs in one big string.
        Useful for creating the tokenizer.
        """
        return "\n".join(self.all_text)

    def __getitem__(self, idx):
        item = torch.tensor(self.all_tokens[idx]).unsqueeze_(0)
        return item

    def __len__(self):
        return len(self.all_tokens)


class WillyShakesDataset(Dataset):

    def __init__(self, tokens, window_size, deterministic=0, size=512):
        self.window_size = window_size
        self.tokens = torch.tensor(tokens).unsqueeze_(0)
        self.n_tokens = len(tokens)

        self.end_idx = self.n_tokens - self.window_size

        self.n_windows = (len(tokens) // window_size) + 1

        self.deterministic = int(deterministic) * 888
        self.size = size

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index beyond dataset size!")

        if self.deterministic:
            random.seed(self.deterministic + idx * 1010101)

        start_idx = random.randint(0, self.end_idx)

        return self.tokens[:, start_idx:start_idx+self.window_size]

    def __len__(self):
        # return self.n_windows
        return self.size


def create_char_tokenizer(data_dir='Metallica_Lyrics', pretrain_tokens={}):

    if data_dir is None:
        all_lyrics = 'a'
    else:
        # Get all of the Lyrics from the dataset
        dataset = MetallicaLyricsDataset(data_dir)
        # and add end of lyrics token
        all_lyrics = dataset.get_all() + ';'

    # This is the essentially the decoder
    all_chars = sorted(list(set(c for c in all_lyrics).union(pretrain_tokens)))

    # This will be the encoder
    token_map = {c:ii for ii,c in enumerate(all_chars)}

    return token_map, all_chars


def clean_willy(all_text):
    # Remove rare chars and a typo
    subs = {'&c': 'etc', '&C': 'etc', '$': 'l'}

    for rem, rep in subs.items():
        all_text = all_text.replace(rem, rep)

    return all_text

def get_shakes_tokens(data_file='shakespeare_input.txt', do_clean=True):
    with open(data_file, 'r', encoding='utf-8') as fp:
        all_text = fp.read()

    if do_clean:
        all_text = clean_willy(all_text)

    char_set = set(c for c in all_text)

    return char_set


@torch.no_grad()
def evaluate(model, dataset, loss_obj, device='cpu'):
    model.eval()
    losses = []
    for song_tokens in dataset:
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

def train(model, dataset, train_params, device='cpu', val_dataset=None, batch_size=16):

    optimizer = AdamW(
        model.parameters(),
        # lr = 3e-4,  # 2e-4
        lr = 1e-3,
        # weight_decay=1e-5,  # 1e-5
    )

    # print the number of parameters in the model
    print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

    torch.manual_seed(1337)

    model.to(device)

    loss_obj = nn.CrossEntropyLoss()
    batch_loss = 0.0

    epoch_losses = []
    n_epochs = train_params.get('n_epochs', 10)

    for e_idx in range(n_epochs):
        losses = []
        for d_idx, song_tokens in enumerate(dataset, 1):
            song_tokens = song_tokens.to(device)
            optimizer.zero_grad()
            output = model(song_tokens)
            labels = song_tokens[:, 1:]

            # Don't use the last output since there is no label
            aligned_output = output[:, :-1, :]
            # Transpose last channel and seq dimension to work with loss
            aligned_output = aligned_output.transpose(-2, -1)

            loss = loss_obj(aligned_output, labels)
            losses.append(loss.item())

            batch_loss = loss + batch_loss

            # Doing an accumulation rather than fancy way.
            # Just to make sure I got architecture correct, then will make fancy
            if (d_idx % batch_size) == 0:
                batch_loss = batch_loss / batch_size
                batch_loss.backward()
                optimizer.step()
                batch_loss = 0.0

        avg_loss = np.mean(losses)
        epoch_losses.append(avg_loss)
        print_str = f'Epoch #{e_idx+1} done! Avg Train Loss: {avg_loss:0.4f}'

        if val_dataset is not None:
            val_loss = evaluate(model, val_dataset, loss_obj, device=device)
            print_str += f' Val loss: {val_loss:0.4f}'

        print(print_str)


@torch.no_grad()
def generate_lyrics(model, title, tokenizer, max_tokens=2000, device='cpu',
                    temp=1.0, top_k=25):

    model.eval()

    lyrics = f'## "{title.upper()}"\n'
    print(lyrics, end="")
    char = ""

    tokens = tokenizer.encode(lyrics)
    tokens = torch.tensor(tokens).unsqueeze_(0)
    tokens = tokens.to(device)

    while char != ';' and tokens.shape[1] < max_tokens:
        output = model(tokens)
        logits = output[:, -1, :] / temp
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = torch.softmax(logits, dim=-1)
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
    parser.add_argument('--split', type=str,
                        help="Split lyrics into 80/20 train/test and copy to this directory")
    parser.add_argument('--pretrain', type=str,
                        help="Use this file to pretrain the model. If ends with .pth, will load it")
    parser.add_argument('--save', type=str, default='metallimore',
                        help='location to store the model. ".pth" will be added to end')

    do_metallica = False

    cli_args = parser.parse_args()
    if cli_args.make_char_tokenizer:
        print('Making the tokenizer!')
        if do_metallica:
            willy_tokens = get_shakes_tokens()
            token_map, all_chars = create_char_tokenizer(
                '~\Dropbox\data\Metallica_Lyrics',
                pretrain_tokens=willy_tokens
            )
            joblib.dump([token_map, all_chars], CHAR_TOKENIZER_FILE)
        else:
            willy_tokens = get_shakes_tokens(do_clean=False)
            token_map, all_chars = create_char_tokenizer(
                None,
                pretrain_tokens=willy_tokens
            )
            print("".join(all_chars))
            print('# of tokens:', len(token_map))
            joblib.dump([token_map, all_chars], SHAKES_TOKENIZER_FILE)
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

    if do_metallica:
        tokenizer = CharTokenizer(CHAR_TOKENIZER_FILE)
    else:
        tokenizer = CharTokenizer(SHAKES_TOKENIZER_FILE)
    n_tokens = len(tokenizer)
    # n_positions = 2048
    # # n_embed = 128
    # n_embed = 384
    # n_heads = 8

    # Andrej's settings:
    # Eval every 100
    # iters = 5000
    # learn rate = 1e-3
    # eval_iters = 200
    n_positions = 32
    n_embed = 64
    n_heads = 4
    n_layers = 4
    batch_size=16


    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


    model = TransCORmer(
        n_tokens, n_embed=n_embed, n_positions=n_positions,
        n_blocks=n_layers, dropout=0.0, n_heads=n_heads
    )

    if cli_args.pretrain:
        if cli_args.pretrain.endswith('.pth'):
            print("Loading pretrained model")
            model = torch.load(cli_args.pretrain)
        else:
            print("Running pre-training!")
            with open(cli_args.pretrain, 'r') as fp:
                text = clean_willy(fp.read())
                tokens = tokenizer.encode(text)
            split_idx = int(len(tokens) * 0.9)

            train_dataset = WillyShakesDataset(
                tokens[:split_idx], window_size=n_positions, size=100*batch_size
            )
            test_dataset = WillyShakesDataset(
                tokens[split_idx:], window_size=n_positions,
                deterministic=888, size=200*batch_size
            )
            train_params = {'n_epochs': 50}
            train(
                model, train_dataset, train_params,
                device=device, val_dataset=test_dataset,
                batch_size=batch_size
            )
            file_name = f'{cli_args.save}.pretrained.pth'
            torch.save(model, file_name)

        import ipdb; ipdb.set_trace()

    # I should probably split in to train and test sets...
    train_dataset = MetallicaLyricsDataset(train_dir, tokenizer)
    test_dataset = MetallicaLyricsDataset(test_dir, tokenizer)

    file_name = f'{cli_args.save}.pth'
    if cli_args.train:
        print('Training Metallimore!')
        train_params = {'n_epochs': 12}
        train(model, train_dataset, train_params, device=device, val_dataset=test_dataset)

        torch.save(model, file_name)
    else:
        model = torch.load(file_name)

    lyrics = generate_lyrics(
        model, title="the forgotten legend", tokenizer=tokenizer, device=device,
        temp=0.8
    )


if __name__ == '__main__':
    main()
