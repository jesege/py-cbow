#!/usr/bin/env python3
import re
from collections import defaultdict, Counter


def read_data(input_file):
    with open(input_file, 'r') as f:
        data = f.read().replace('\n', ' ').lower()

    return data


def normalize(data):
    result = []
    pattern = re.compile("[^\W\d_]+")
    sentences = re.split("[.?!]", data)
    word_occurences = dict(Counter(re.findall(pattern, data)))
    for s in sentences:
        sentence_list = [w for w in re.findall(pattern, s)
                         if word_occurences.get(w) > 10]
        if len(sentence_list) > 4:
            result.append(' '.join(sentence_list))

    return result


def vocabulary():
    dictionary = defaultdict()
    dictionary.default_factory = lambda: len(dictionary)
    return dictionary


def docs2bow(docs, dictionary):
    """Transforms a list of strings into a list of lists where
    each unique item is converted into a unique integer."""
    for doc in docs:
        yield [dictionary[word] for word in doc.split()]


def main(input_file):
    data = read_data(input_file)
    sentences = normalize(data)
    vocab = Vocabulary()
    sentences_bow = list(docs2bow(sentences, vocab))
    print(len(vocab))

if __name__ == '__main__':
    main('data/svwiki.txt')
