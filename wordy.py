#!/usr/bin/env python3

from collections import defaultdict
from random import randint
import numpy as np
import pandas as pd


embeddings_filename = '.embeddings.feather'
print('loading... ', flush=True, end='')
good_words = set(w.strip() for w in open('relevant-words.txt', encoding='utf8'))
df = pd.read_feather(embeddings_filename)
embeddings = [(word,em) for word,em in zip(df.word, df.embedding) if word in good_words]
print('done')


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


while True:
    i = randint(0, len(embeddings)-1)
    word,em = embeddings[i]
    print()
    print()
    # print(word)
    # print('-'*len(word))

    closest = sorted(embeddings, key=lambda kv:-cosine_similarity(em, kv[1]))
    pos = {word_em[0]:i for i,word_em in enumerate(closest)}
    # for word,diff in closest[:20]:
        # print(word)

    i = 1
    guesses = {}
    while True:
        guess = input(f'Guess {i}: ')
        if not guess:
            print()
            for guess,score in sorted(guesses.items(), key=lambda gs:gs[1]):
                print(guess, '=', score)
            print()
            continue
        if guess == '?':
            print()
            for i,(word,em) in enumerate(closest[:100], 1):
                print(word, '=', i)
            print()
            continue
        if guess not in pos:
            print('bad word')
            continue
        score = pos[guess]
        guesses[guess] = score
        print(guess, '=', score)
        i += 1
