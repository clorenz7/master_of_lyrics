import argparse
import glob
import os
import re

import joblib

from torch import nn
from torch.utils.data import Dataset

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


def main():
    parser = argparse.ArgumentParser(
        'Script for creating a Metallica Lyric generator'
    )
    parser.add_argument(
        'data_dir', metavar='data_dir',
        help='Directory containing files with .txt lyrics files'
    )
    parser.add_argument('--make_char_tokenizer', action='store_true')

    cli_args = parser.parse_args()
    if cli_args.make_char_tokenizer:
        token_map, all_chars = create_char_tokenizer()
        joblib.dump([token_map, all_chars], CHAR_TOKENIZER_FILE)

    tokenizer = CharTokenizer(CHAR_TOKENIZER_FILE)

    dataset = MetallicaLyricsDataset(cli_args.data_dir, tokenizer)

    print(dataset.all_text[0])
    print(tokenizer.decode(dataset[0]))


if __name__ == '__main__':
    main()
