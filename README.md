# Metalli-More
"Master of Lyrics I'm Transforming Your Tokens
Learning your patterns and sampling out new verses
Reading from me, you can't tell a difference
Just give me a title, I'll make you new lyrics"

Inspired by Makemore by Andrej Karpathy. But instead of generating the works of Shakespeare, we generate Metallica lyrics.


## Usage

You can use the metallimore.py script to train a Transformer to generate Metallica lyrics given only a directory containing song lyrics in .txt files:

```bash
python metallimore.py ~/metallica_lyrics --split ~/metallica_split/ --pretrain ./shakespeare_input.txt --make_char_tokenizer  --train --save metallimore_test -d 0.3 --eval 'master of lyrics,transformer'
```

This will split the songs into train and validation sets, pretrain on a shakespeare corpus, make a character tokenizer, save the results, and run the model to generate songs with titles of "Master of Lyrics" and "Transformer". All with a 30% dropout rate.

After running this, New songs can be generated with:

```bash
python metallimore.py ~/metallica_split --save metallimore_test --eval 'wrecking ball, poker face,firework'
```

You can use the [willy_shakes.py](./willy_shakes.py) script to download the pretraining data.