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

    # A dict of word:sortings which will contain every word in the word list and list of cosine similarities to the guesses.
    # For example: { 'horse': [0.2, 0.7, 0.1] }
    similarities = defaultdict(list)
    guess_score = list(zip(guesses[::2], guesses[1::2]))
    desired_score = []
    for guess,score in guess_score:
        guess_embedding = embeddings.get(guess)
        if guess_embedding is None:
            print(f'skipping {guess}')
            continue
        desired_score.append(int(score))
        for word,embedding in zip(df.word, df.embedding):
            if word == guess:
                continue
            sim = cosine_similarity(embedding, guess_embedding)
            similarities[word].append(sim)

    desired_score = np.array([s/100_000 for s in desired_score])
    weight = np.array([1/(i+1) for i in range(len(desired_score))])
    similarities = {k:np.array(v) for k,v in similarities.items() if len(v) == len(desired_score)}
    dot2 = lambda x: np.dot(x,x)
    similarities = sorted(similarities.items(), key=lambda kv:-dot2(((desired_score-kv[1])*weight)))
    for word,sim in similarities[:20]:
        print(word)
