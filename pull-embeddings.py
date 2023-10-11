#!/usr/bin/env python3

import pandas as pd
import openai
import os


openai.api_key = os.getenv('OPENAI_API_KEY')
embeddings = {}
embeddings_filename = '.embeddings.feather'
words_filename = '/usr/share/dict/words'
skip_save_counter = 0
registered = set()


if not openai.api_key:
    print('you need to a) create an OpenAI account, b) create API keys, and c) set the OPENAI_API_KEY system variable')
    exit(1)


def filter_words(l):
    for w in l:
        w = w.rstrip()
        if w[-1] == 's' and len(w) > 3:
            if w[:-1] in registered:
                continue
        if w[-2:] == 'ed' and len(w) > 4:
            continue
        if w[-3:] == 'ing' and len(w) > 5:
            continue
        if any(ch<'a' or ch>'z' for ch in w): # only lowercase letters
            continue
        registered.add(w)
        yield w


def chunks(l, chunk_len):
    r = []
    for e in l:
        r.append(e)
        if len(r) >= chunk_len:
            yield r
            r.clear()
    if r:
        yield r


def update_embeddings(fname=None, chunk_len=2000):
    if fname is None:
        fname = words_filename
    for words in chunks(filter_words(open(fname)), chunk_len):
        words = [w for w in words]
        update_embeddings_from_words(words)
    save_embeddings()


def update_embeddings_from_words(words):
    global skip_save_counter
    load_embeddings()
    words = [w for w in words if w not in embeddings]
    if not words:
        return
    print(words[0])
    for word,embedding in zip(words, _query_embeddings(words)):
        embeddings[word] = embedding
    skip_save_counter += 1
    if skip_save_counter % 5 == 0:
        save_embeddings()


def load_embeddings():
    global embeddings
    if not embeddings:
        try:
            df = pd.read_feather(embeddings_filename)
            embeddings = {word:embedding for word,embedding in zip(df.word, df.embedding)}
        except:
            pass


def save_embeddings():
    word = []
    embedding = []
    for k,v in embeddings.items():
        word.append(k)
        embedding.append(v)
    df = pd.DataFrame({'word':word, 'embedding':embedding})
    df.to_feather(embeddings_filename, compression='zstd')


def _query_embeddings(words, engine='text-similarity-babbage-001'):
    assert len(words) <= 2048, 'The batch size should not be larger than 2048.'
    words = [word.replace('\n', ' ') for word in words]
    data = openai.Embedding.create(input=words, engine=engine).data
    data = sorted(data, key=lambda x: x['index'])
    return [d['embedding'] for d in data]


if __name__ == '__main__':
    update_embeddings()
