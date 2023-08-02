import glob
import os
import re

import joblib
import torch
from torch.utils.data import Dataset


def remove_bracketed_text(text):
    """
    Removes comments like [solo] or [Chorus] from song lyrics
    """
    pattern = r"\s*\[.*?\]\s*"
    cleaned_text = re.sub(pattern, '', text)
    return cleaned_text


def get_title(text):
    """
    Obtains TITLE from string format '## "{TITLE}"'
    """
    pattern = r'## "(.+)"'

    match = re.search(pattern, text)

    if match:
        title = match.group(1)
    else:
        raise ValueError("No title found!")

    return title


def clean_lyrics(all_lyrics):
    """
    Replace rare and weird characters found in Metallica lyrics files.
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
    all_lyrics = all_lyrics.replace('  ', ' ')
    all_lyrics = all_lyrics.replace('(', '')
    all_lyrics = all_lyrics.replace(')', '')

    return all_lyrics.strip()


class CharTokenizer(object):
    """
    Simple character level tokenizer class ala Hugging Face
    """

    def __init__(self, token_map_file, eos_token=';', pad_token=' '):
        self.encode_map, self.decode_list = joblib.load(token_map_file)
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.pad_token_id = self.encode_map[pad_token]
        self.eos_token_id = self.encode_map[eos_token]

    def encode(self, text):
        return [self.encode_map[c] for c in text]

    def decode(self, tokens):
        return "".join(self.decode_list[t] for t in tokens)

    def __len__(self):
        return len(self.decode_list)


class MetallicaLyricsDataset(Dataset):
    """
    Dataset class for a folder of Metallica lyrics
    """

    _tokenizer_class_map = {
        'char_tokenizer': CharTokenizer
    }

    def __init__(self, data_dir, tokenizer=None, window_size=0,
                 cat_mode=True, size=None, reformat_title=True):
        """
        Args:
            data_dir:
                Directory with all of the songs in separate .txt files
            tokenizer:
            window_size:
                How many characters to grab at a time
            cat_mode:
                If True, lyrics from all songs will be concatentated together
            size:
                How many windows to take from the data
            reformat_title:
                Converts first line from '## "TITLE"' to 'TITLE:'
        """
        file_list = glob.glob(
            os.path.join(os.path.expanduser(data_dir), '*.txt')
        )

        self.all_text = []
        self.all_tokens = []

        self.tokenizer = tokenizer

        self.window_size = window_size

        self.cat_mode = cat_mode

        # Read files from the directory
        for file_name in file_list:
            with open(file_name, 'r') as fp:
                song_text = clean_lyrics(fp.read())
                if reformat_title:
                    title, song_text = song_text.split('\n', 1)
                    title = get_title(title)
                    song_text = f'{title}:\n' + song_text
                self.all_text.append(song_text)

            if not self.cat_mode:
                if tokenizer is not None:
                    self.all_tokens.append(
                        tokenizer.encode(song_text + tokenizer.eos_token)
                    )
        if self.cat_mode:
            eos_token = '' if tokenizer is None else tokenizer.eos_token
            all_songs = self.get_all(sep=eos_token+'\n')
            all_songs += eos_token
            self.all_tokens = tokenizer.encode(all_songs)
            self.end_idx = len(self.all_tokens) - self.window_size
            self.size = size or 1600
        else:
            self.size = size or len(self.all_tokens)

    def get_all(self, sep="\n"):
        """
        Returns the lyrics from all songs in one big string.
        Useful for creating the tokenizer.
        """
        return sep.join(self.all_text)

    def __getitem__(self, idx):

        if self.cat_mode:
            start_idx = torch.randint(0, self.end_idx, (1,))

            item = torch.tensor(
                self.all_tokens[start_idx:start_idx+self.window_size]
            )
        else:
            song_idx = torch.randint(0, len(self.all_tokens), (1,))
            item = torch.tensor(self.all_tokens[song_idx])

            if self.window_size:
                n_tokens = item.shape[0]
                if n_tokens < self.window_size:
                    tokens = item
                    item = torch.full(
                        (self.window_size,),
                        self.tokenizer.pad_token_id,
                        dtype=torch.long
                    )
                    item[:n_tokens] = tokens
                else:
                    start_idx = torch.randint(-5, n_tokens - self.window_size + 5, (1,))
                    start_idx = min(start_idx, n_tokens - self.window_size)
                    start_idx = max(start_idx, 0)

                    item = item[start_idx:start_idx+self.window_size]

        return item
        # return item.unsqueeze_(0)

    def __len__(self):
        return self.size


class WillyShakesDataset(Dataset):
    """
    Dataset class for Karpathy's concatenated shakespeare file
    """

    def __init__(self, tokens, window_size, deterministic=0, size=512):
        self.window_size = window_size
        self.tokens = torch.tensor(tokens)

        self.n_tokens = len(tokens)

        self.end_idx = self.n_tokens - self.window_size

        self.n_windows = (len(tokens) // window_size) + 1

        self.deterministic = int(deterministic) * 888
        self.size = size

    def __getitem__(self, idx):

        # if self.deterministic:
        #     random.seed(self.deterministic + idx * 1010101)

        start_idx = torch.randint(0, self.end_idx, (1,))
        start_idx = start_idx.item()

        return self.tokens[start_idx:start_idx+self.window_size]

    def __len__(self):
        return self.size


def create_char_tokenizer(data_dir='Metallica_Lyrics', pretrain_tokens={}):

    if data_dir is None:
        all_lyrics = 'a'
    else:
        # Get all of the Lyrics from the dataset
        dataset = MetallicaLyricsDataset(data_dir, cat_mode=False)
        # and add end of lyrics token
        all_lyrics = dataset.get_all() + ';'

    # This is the essentially the decoder
    all_chars = sorted(list(set(c for c in all_lyrics).union(pretrain_tokens)))

    # This will be the encoder
    token_map = {c: ii for ii, c in enumerate(all_chars)}

    return token_map, all_chars


def clean_willy(all_text):
    """
    Removes rare chars and a typo from the concatenated shakespere file
    """
    subs = {'&c': 'etc', '&C': 'etc', '$': 'l'}

    for rem, rep in subs.items():
        all_text = all_text.replace(rem, rep)

    return all_text


def get_shakes_tokens(data_file='shakespeare_input.txt', do_clean=True):
    """
    Obtains all of the characters in the shakespere file
    Args:
        do_clean:
            If True, the file will be corrected before processing.
            False is useful for replicating Andrej's results
    """
    with open(data_file, 'r', encoding='utf-8') as fp:
        all_text = fp.read()

    if do_clean:
        all_text = clean_willy(all_text)

    char_set = set(c for c in all_text)

    return char_set
