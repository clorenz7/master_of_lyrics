import requests
import os


if __name__ == '__main__':
    input_file_path = os.path.join(os.path.dirname(__file__), 'shakespeare_input.txt')
    if not os.path.exists(input_file_path):
        data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
        with open(input_file_path, 'w') as f:
            f.write(requests.get(data_url).text)
