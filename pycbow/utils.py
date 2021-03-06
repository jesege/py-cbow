#!/usr/bin/env python3
import re
from collections import defaultdict, Counter
import numpy as np


def read_data(input_file):
    with open(input_file, 'r') as f:
        data = f.read().replace('\n', ' ').lower()

    return data


def normalize(data, min_word_occurences=5, min_sentence_length=10):
    result = []
    pattern = re.compile("[^\W\d_]+")
    sentences = re.split("[.?!]", data)
    word_occurences = dict(Counter(re.findall(pattern, data)))
    for s in sentences:
        if len(s) > min_sentence_length:
            sentence_list = [w for w in re.findall(pattern, s)
                             if word_occurences.get(w) > min_word_occurences]
            result.append(' '.join(sentence_list))

    return result


def subsample(word_occurences, t=10):
    removed_words = []
    for word, freq in word_occurences.items():
        prob = 1 - (np.sqrt(t/freq))
        rand = np.random.random(1)[0]
        if prob > rand:
            removed_words.append(word)
    return removed_words


def vocabulary():
    dictionary = defaultdict()
    dictionary.default_factory = lambda: len(dictionary)
    return dictionary


def docs2bow(docs, dictionary):
    """Transforms a list of strings into a list of lists where
    each unique item is converted into a unique integer."""
    for doc in docs:
        yield [dictionary[word] for word in doc.split()]
