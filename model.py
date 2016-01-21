"""
A simple implementation of the continuous bag of words model
from word2vec. Code is based on [1].

[1] http://www.folgertkarsdorp.nl/word2vec-an-introduction/
"""
import numpy as np
from numpy import linalg as la
from collections import defaultdict
import utils
import pickle


class CBOW(object):

    def __init__(self, vocabulary, rng, hidden_size=100,
                 learning_rate=0.1):
        # The vocabulary used in the model
        self.vocabulary = vocabulary
        self.id2word = {v: k for k, v in vocabulary.items()}
        # V is the number of words in our vocabulary
        self.V = len(vocabulary)
        # N is a hyperparameter of the model, specifying
        # the number of nodes in the hidden layer and resulting
        # dimensionality of our vectors
        self.N = hidden_size
        # syn0 is our weights from input layer to projection layer,
        # initialized to random numbers
        self.syn0 = np.asarray(
                            rng.uniform(
                                low=-(1/(2*self.N)),
                                high=(1/(2*self.N)),
                                size=(self.V, self.N)),
                            dtype=np.float64)
        # syn1 is our weights from hidden layer to output layer
        self.syn1 = np.asarray(
                            rng.uniform(
                                low=-(1/(2*self.N)),
                                high=(1/(2*self.N)),
                                size=(self.N, self.V)),
                            dtype=np.float64)
        self.learning_rate = learning_rate

    def update(self, target_word, context):
        """Update the weights of this model given a training sample
        (a word and its context)."""
        # Accumulate sum of output vectors weighted by prediction error
        eh = np.zeros(self.N)
        # "Hidden layer" transformation
        h = (1.0 / len(context)) * (sum(self.syn0[c] for c in context))
        # Compute output probabilities
        div = sum(np.exp(np.dot(self.syn1.T[w], h))
                  for w in self.vocabulary.values())
        probabilities = np.exp(np.dot(self.syn1.T, h)) / div

        for index, p in enumerate(probabilities):
            t = 1 if target_word == index else 0
            error = p - t

            self.syn1.T[index] = (self.syn1.T[index] -
                                  self.learning_rate * error * h)

            eh += error * self.syn1.T[index]

        for c in context:
            self.syn0[c] = (self.syn0[c] - (1 / len(context)) *
                            self.learning_rate * eh)

    def most_similar(self, word):
        """Calculate the most similar word using the output
        matrix (syn1) vectors with cosine similarity.
        """
        if isinstance(word, str):
            word = self.vocabulary.get(word)

        most_similar = 0
        highest_similarity = 0
        for i in self.vocabulary.values():
            if i != word:
                cos_sim = cosine_similarity(self.syn1.T[word], self.syn1.T[i])
                if cos_sim > highest_similarity:
                    highest_similarity = cos_sim
                    most_similar = i

        return self.id2word.get(most_similar), highest_similarity


def cosine_similarity(u, v):
    """Return the cosine similarity of two vectors."""
    sim = np.dot(u, v) / (la.norm(u) * la.norm(v))
    return sim


def train(cbow_model, sentences):
    """Update the weights of a continuous-bag-of-words
    model with the given sentences."""
    no_sentences = len(sentences)
    i = 0
    for s in sentences:
        print("Training on sentence {0} ({1} words).".format(
                                        no_sentences - i, len(s)))
        i = i + 1
        training_instances = build_context(s)
        for inst in training_instances:
            focus_word = inst[0]
            context = inst[1]
            cbow_model.update(focus_word, context)


def build_context(sentence):
    """For a sentence, build the training instances (word and
    context pairs)."""
    # this code is pretty ugly and should be rewritten so it can be read
    # and used for arbitrary contexts
    i = 0
    training_instances = []
    for word in sentence:
        context_words = []
        if i == 0:
            context_words.extend(sentence[1:3])
        elif i == 1:
            context_words.append(sentence[i - 1])
            context_words.extend(sentence[i + 1:4])
        elif i == len(sentence) - 2:
            context_words.extend(sentence[-4:-2])
            context_words.extend(sentence[-1:])
        elif i == len(sentence) - 1:
            context_words.extend(sentence[-3:-2])
        else:
            context_words.extend(sentence[i - 2:i])
            context_words.extend(sentence[i+1:i + 3])

        instance = [word, tuple(context_words)]
        training_instances.append(instance)
        i = i + 1

    return training_instances


def main(input_file):

    print("Reading data...")
    data = utils.normalize(utils.read_data(input_file))
    vocab = utils.vocabulary()
    sentences = list(utils.docs2bow(data, vocab))
    rng = np.random.RandomState(123)
    print("Data loaded!")
    print("{0} words in the vocabulary.".format(len(vocab)))
    print("Starting training.")
    model = CBOW(vocab, rng)
    train(model, sentences)


if __name__ == '__main__':
    main("data/svwiki.txt")
