#!/usr/bin/env python3

for l in open('good-words.txt', encoding='utf8'):
    _,w,_ = l.split('\t')
    if all(ch>='A' and ch<='z' for ch in w):
        print(w.lower())
