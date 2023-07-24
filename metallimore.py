import argparse
import glob
import os
import random
import re
import shutil

import joblib

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
        file_list = glob.glob(os.path.join(data_dir, '*.txt'))

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
        return self.all_tokens[idx]

    def __len__(self):
        return len(self.all_tokens)


def create_char_tokenizer(data_dir='Metallica_Lyrics'):
    # Get all of the Lyrics from the dataset
    dataset = MetallicaLyricsDataset(data_dir)
    all_lyrics = dataset.get_all()

    # This is the essentially the decoder
    all_chars = sorted(list(set(c for c in all_lyrics)))
    # Add an End of Song token
    all_chars += ';'

    # This will be the encoder
    token_map = {c:ii for ii,c in enumerate(all_chars)}

    return token_map, all_chars


class TransCORmer(nn.Module):

    def __init__(self, n_tokens, n_embed, n_positions=2048):
        super().__init__()
        # Create position and word embeddings
        self.token_embed = nn.Embedding(n_tokens, n_embed)
        self.pos_embed = nn.Embedding(n_positions, n_embed)

    def forward(self, x):

        e = self.token_embed(x) + self.pos_embed(x)

        return e

def train(model, dataset, train_params):

    optimizer = AdamW(
        model.parameters(),
        lr = 2e-4,
        weight_decay=1e-5,
    )

    loss_obj = nn.CrossEntropyLoss()

    for tokens in dataset:
        optimizer.zero_grad()

        output = model(tokens)
        loss = loss_obj(output, tokens)

        loss.backward()
        optimizer.step()




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
                        help="Split into 80/20 train/test and copy to this directory")

    cli_args = parser.parse_args()
    if cli_args.make_char_tokenizer:
        token_map, all_chars = create_char_tokenizer()
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
    n_embed = 128

    model = TransCORmer(n_tokens, n_embed=n_embed, n_positions=n_positions)

    # I should probably split in to train and test sets...
    train_dataset = MetallicaLyricsDataset(train_dir, tokenizer)
    test_dataset = MetallicaLyricsDataset(test_dir, tokenizer)

    if cli_args.train:
        train_params = {}
        train(model, train_dataset, train_params)


if __name__ == '__main__':
    main()
