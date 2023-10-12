#!/usr/bin/env python3

from collections import defaultdict
import numpy as np
import pandas as pd


embeddings_filename = '.embeddings.feather'


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


df = pd.read_feather(embeddings_filename)

while True:
    embeddings = {word:embedding for word,embedding in zip(df.word, df.embedding)}
    print('\nPaste guesses, terminate with empty line>')
    guesses = []
    while True:
        g = input()
        if not g:
            break
        guesses += [g]

    print('Processing...\n')

    # A dict of word:sortings which will contain every word in the word list and list of cosine similarties to the guesses.
    # For example: { 'horse': [0.2, 0.7, 0.1] }
    diffs = defaultdict(list)
    guess_rank = list(zip(guesses[::2], guesses[1::2]))
    desired_rank = []
    for guess,rank in guess_rank:
        guess_embedding = embeddings.get(guess)
        if guess_embedding is None:
            print(f'skipping {guess}')
            continue
        desired_rank.append(int(rank))
        for word,embedding in zip(df.word, df.embedding):
            if word == guess:
                continue
            diff = 1 - cosine_similarity(embedding, guess_embedding)
            diffs[word].append(diff)

    desired_rank = np.array([(r/100_000)**(1/2) for r in desired_rank])
    weight = np.array([1/(i+5) for i in range(len(desired_rank))])
    diffs = {k:np.array(v) for k,v in diffs.items() if len(v) == len(desired_rank)}
    dot2 = lambda x: np.dot(x,x)
    diffs = sorted(diffs.items(), key=lambda kv:dot2((desired_rank-kv[1])*weight))
    for word,diff in diffs[:20]:
        print(word)
