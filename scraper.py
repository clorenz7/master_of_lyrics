
import os
import random
import re
import time


from bs4 import BeautifulSoup
import joblib
import requests


BASE_URL = 'https://www.azlyrics.com'
MAIN_URL = 'https://www.azlyrics.com/m/metallica.html'

SONG_LINK_REGEX = r'h:"(.*?\.html)"'

def get_song_links(cache_file="metallica_songs.joblib"):

    if os.path.exists(cache_file):
        return joblib.load(cache_file)

    song_links = []

    resp = requests.get(MAIN_URL)

    soup = BeautifulSoup(resp.text, "html.parser")
    # print(soup.prettify())
    script_tags = soup.find_all("script")

    song_var_list = [ t.text for t in script_tags if 's:"Master Of Puppets"' in t.text]

    song_var_text = song_var_list[0].split('var songlist = [', 1)[1]
    song_var_text = song_var_text.split("];", 1)[0].strip()

    songs = song_var_text.split('\n')

    for song_line in songs:
        # Find all matches of the regex in the input string
        matches = re.findall(SONG_LINK_REGEX, song_line)

        if matches:
            link = BASE_URL + matches[0]
            song_links.append(link)

    joblib.dump(song_links, cache_file)

    return song_links


def main():

    out_dir = 'Metallica_Lyrics'

    song_links = get_song_links()

    songs_by_title = {}

    for song_link in song_links:
        sani_title = os.path.splitext(os.path.basename(song_link))[0]
        out_file = os.path.join(out_dir, sani_title + '.txt')

        if os.path.exists(out_file):
            continue

        time.sleep(290 + 20*random.random())
        resp = requests.get(song_link)
        resp_text = resp.content.decode('utf-8', 'ignore')

        soup = BeautifulSoup(resp_text, "html.parser")
        ringtone_tag = soup.find("div", class_="ringtone")

        next_sibs = ringtone_tag.fetchNextSiblings()
        title = ''

        for tag in next_sibs:
            if tag.name == 'b' and title == '':
                title = tag.text
                print(f'Found "{title}"')
            if tag.name == 'br':
                continue

            if tag.name == 'div':
                lyrics = tag.text
                lyrics = lyrics.replace('’', "'").strip()
                songs_by_title[title] = lyrics
                break

        with open(out_file, 'w') as fp:
            fp.write(f'## {title.upper()}')
            fp.write('\n')
            fp.write(lyrics)
            fp.write('\n')
        print(f"Wrote lyrics to {out_file}")



if __name__ == '__main__':
    main()