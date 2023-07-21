import glob
import os
import re

from torch import nn
from torch.utils.data import Dataset


# TODO: Use tiktoken to train a SimpleBytePairEncoding on the data corpus

def remove_bracketed_text(text):
    # Define the regular expression pattern
    pattern = r"\s*\[.*?\]\s*"

    # Use re.sub() to remove the matched pattern from the text
    cleaned_text = re.sub(pattern, '', text)

    return cleaned_text

def clean_lyrics(all_lyrics):
    """
    Remove rare and weird characters.
    Should get down to 70 character vocabulary.
    """

    all_lyrics = remove_bracketed_text(all_lyrics)

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
    all_lyrics = all_lyrics.replace('  ', '\n')
    all_lyrics = all_lyrics.replace('(', '')
    all_lyrics = all_lyrics.replace(')', '')

    return all_lyrics


class MetallicaLyricsDataset(Dataset):

    def __init__(self, data_dir, tokenizer=None):
        file_list = glob.glob(os.path.join(data_dir, '*.txt'))

        self.all_text = []

        for file_name in file_list:
            with open(file_name, 'r') as fp:
                song_text = fp.read()
                self.all_text.append(clean_lyrics(song_text))

    def get_all(self):
        return "\n".join(self.all_text)


def create_char_tokenizer(data_dir='Metallica_Lyrics'):
    dataset = MetallicaLyricsDataset(data_dir)
    all_lyrics = dataset.get_all()

    all_chars = sorted(list(set(c for c in all_lyrics)))

    token_map = {c:ii for ii,c in enumerate(all_chars)}

    return token_map



def main():
    token_map = create_char_tokenizer()


if __name__ == '__main__':
    main()
