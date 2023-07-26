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
from torch.utils.data import Dataset
from torch.optim import AdamW

CHAR_TOKENIZER_FILE = 'char_tokenizer.joblib'


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

    def __init__(self, tokens, window_size, deterministic=0):
        self.window_size = window_size
        self.tokens = torch.tensor(tokens).unsqueeze_(0)
        self.n_tokens = len(tokens)

        self.end_idx = self.n_tokens - self.window_size

        self.n_windows = (len(tokens) // window_size) + 1

        self.deterministic = int(deterministic) * 888

    def __getitem__(self, idx):
        if idx >= len(self):
            raise IndexError("Index beyond dataset size!")

        if self.deterministic:
            random.seed(self.deterministic + idx * 1010101)

        start_idx = random.randint(0, self.end_idx)

        return self.tokens[:, start_idx:start_idx+self.window_size]

    def __len__(self):
        # return self.n_windows
        return 512



def create_char_tokenizer(data_dir='Metallica_Lyrics', pretrain_tokens={}):
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

def get_shakes_tokens(data_file='shakespeare_input.txt'):
    with open(data_file, 'r') as fp:
        all_text = fp.read()

    all_text = clean_willy(all_text)

    char_set = set(c for c in all_text)

    return char_set


class AttentionHead(nn.Module):

    def __init__(self, n_embed_in, n_embed_out, max_tokens=2048, dropout=0.2):
        super().__init__()
        self.n_embed = n_embed_in

        self.query_xform = nn.Linear(n_embed_in, n_embed_out, bias=False)
        self.key_xform = nn.Linear(n_embed_in, n_embed_out, bias=False)
        self.value_xform = nn.Linear(n_embed_in, n_embed_out, bias=False)

        self.dropout = nn.Dropout(dropout)

        self.register_buffer('no_look_ahead', torch.triu(torch.full((max_tokens, max_tokens), float('-inf')), diagonal=1))

    def forward(self, x):
        """
        expected shape:
            batch x tokens x embed
        """
        Q = self.query_xform(x)
        K = self.key_xform(x).transpose(-2, -1)
        V = self.value_xform(x)

        n_tokens = x.shape[1]

        NLA = self.no_look_ahead[:n_tokens, :n_tokens]

        dk = math.sqrt(Q.shape[-1])

        attention = torch.softmax((Q @ K)/dk + NLA, dim=2)
        attention = self.dropout(attention)

        output = attention @ V

        return output


class MultiHeadAttention(nn.Module):

    # def __init__(self, n_embed, n_heads=8, n_inner=512, dropout=0.2):
    def __init__(self, n_embed, n_heads=6, n_inner=256, dropout=0.2):
        super().__init__()
        self.n_heads = n_heads
        self.n_embed = n_embed

        dim_per_head = n_embed // n_heads

        self.heads = nn.ModuleList(
            [AttentionHead(n_embed, dim_per_head, dropout=dropout) for _ in range(n_heads)]
        )

        self.fc_layer = nn.Sequential(
            nn.Linear(n_embed, n_inner),
            nn.ReLU(),
            nn.Linear(n_inner, n_embed),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        ## Implement and Add Layer Normalization

        attentions = [head(x) for head in self.heads]

        y = torch.concat(attentions, dim=2)
        y = self.fc_layer(y)
        # Add residual
        y = y + x

        return y


class TransCORmer(nn.Module):

    # TODO: Add dropout

    def __init__(self, n_tokens, n_embed, n_blocks=4, n_positions=2048,
                 dropout=0.2):
        super().__init__()
        # Create position and word embeddings
        self.token_embed = nn.Embedding(n_tokens, n_embed)
        self.pos_embed = nn.Embedding(n_positions, n_embed)

        self.attention_blocks = nn.Sequential(
            *[MultiHeadAttention(n_embed, dropout=dropout) for _ in range(n_blocks)]
        )

        self.dropout = nn.Dropout(dropout)

        self.lm_head = nn.Linear(n_embed, n_tokens)

    def forward(self, x):

        e = self.token_embed(x) + self.pos_embed(x)
        y = self.attention_blocks(e)

        y = self.dropout(y)

        y = self.lm_head(y)
        # I think the loss function will do this.
        # y = torch.softmax(y, dim=2)

        return y

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

def train(model, dataset, train_params, device='cpu', val_dataset=None):

    optimizer = AdamW(
        model.parameters(),
        lr = 3e-4,  # 2e-4
        weight_decay=1e-5,  # 1e-5
    )

    model.to(device)

    loss_obj = nn.CrossEntropyLoss()

    epoch_losses = []
    n_epochs = train_params.get('n_epochs', 10)

    for e_idx in range(n_epochs):
        losses = []
        for song_tokens in dataset:
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

            loss.backward()
            optimizer.step()

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

    cli_args = parser.parse_args()
    if cli_args.make_char_tokenizer:
        print('Making the tokenizer!')
        willy_tokens = get_shakes_tokens()
        token_map, all_chars = create_char_tokenizer(
            '~\Dropbox\data\Metallica_Lyrics',
            pretrain_tokens=willy_tokens
        )
        joblib.dump([token_map, all_chars], CHAR_TOKENIZER_FILE)
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

    tokenizer = CharTokenizer(CHAR_TOKENIZER_FILE)
    n_positions = 2048
    n_tokens = len(tokenizer)
    # n_embed = 128
    n_embed = 384

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    model = TransCORmer(
        n_tokens, n_embed=n_embed, n_positions=n_positions,
        # n_blocks=1, dropout=0.0,
        n_blocks=2, dropout=0.2
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
                tokens[:split_idx], window_size=n_positions
            )
            test_dataset = WillyShakesDataset(
                tokens[split_idx:], window_size=n_positions,
                deterministic=888
            )
            train_params = {'n_epochs': 10}
            train(
                model, train_dataset, train_params,
                device=device, val_dataset=test_dataset
            )
            file_name = f'{cli_args.save}.pretrained.pth'
            torch.save(model, file_name)


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
