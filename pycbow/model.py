"""
A simple implementation of the continuous bag of words model
from word2vec.
"""
import numpy as np
from numpy import linalg
from pycbow import utils
import sys


class CBOW(object):

    def __init__(self, vocabulary, rng, hidden_size=50,
                 learning_rate=0.025):
        # The vocabulary used in the model
        self.word2id = vocabulary
        self.id2word = {v: k for k, v in vocabulary.items()}
        self.word_ids = sorted([v for v in vocabulary.values()])
        # V is the number of words in our vocabulary
        self.V = len(vocabulary)
        # hidden_size is the desired dimensionality of the word vectors
        self.N = hidden_size
        # syn0 is our weights from input layer to projection layer,
        # initialized to random numbers
        self.syn0 = np.asarray(
            rng.uniform(
                low=-(1/(2*self.N)),
                high=(1/(2*self.N)),
                size=(self.V, self.N)),
            dtype=np.float64)
        # syn1 is our output vectors
        self.syn1 = np.asarray(
            rng.uniform(
                low=-(1/(2*self.N)),
                high=(1/(2*self.N)),
                size=(self.N, self.V)),
            dtype=np.float64)
        self.eta = learning_rate

    def update(self, target_word, context):
        """Update the weights of this model given a training sample
        (a word and its context)."""
        # Accumulate sum of output vectors weighted by prediction error
        eh = np.zeros(self.N)
        # Projection layer transformation
        h = np.mean(np.array([self.syn0[c] for c in context]), axis=0)
        # Output transformation
        exp_dot = np.exp(np.dot(self.syn1.T, h))
        probabilities = exp_dot / sum(exp_dot)

        for word, y in enumerate(probabilities):
            t = 1 if target_word == word else 0
            error = y - t
            eh += error * self.syn1.T[word]

            self.syn1.T[word] -= self.eta * error * h

        for c in context:
            self.syn0[c] -= (1 / len(context)) * (self.eta * eh)

    def most_similar(self, word):
        """Calculate the most similar word using the output
        matrix (syn1) vectors with cosine similarity.
        """
        if isinstance(word, str):
            word = self.word2id.get(word)

        if word is None:
            return None

        most_similar = 0
        highest_similarity = 0
        for i in self.word_ids:
            if i == word:
                continue

            cos_sim = cosine_similarity(self.syn1.T[word], self.syn1.T[i])
            if cos_sim > highest_similarity:
                highest_similarity = cos_sim
                most_similar = i

        return self.id2word.get(most_similar), highest_similarity


def cosine_similarity(u, v):
    """Return the cosine similarity of two vectors."""
    sim = np.dot(u, v) / (linalg.norm(u) * linalg.norm(v))
    return sim


def train(cbow_model, sentences, window_size, epochs):
    """Update the weights of a continuous-bag-of-words
    model with the given sentences."""
    for i in range(epochs):
        for no_sentences, s in enumerate(sentences):
            if no_sentences % 10 == 0:
                sys.stdout.write('\rSentence {0} of {1}'.format(
                    no_sentences, len(sentences)))
                sys.stdout.flush()

            training_instances = build_context(s, window_size)
            for inst in training_instances:
                focus_word = inst[0]
                context = inst[1]
                cbow_model.update(focus_word, context)


def build_context(sentence, window_size):
    """For a sentence, build the training instances (word and
    context pairs)."""
    training_instances = []
    for i, word in enumerate(sentence):
        context_start = max(i - window_size, 0)
        context_end = min(i + window_size + 1, len(sentence))
        context_words = sentence[context_start:i] + sentence[i+1:context_end]

        if context_words:
            instance = [word, tuple(context_words)]
            training_instances.append(instance)

    return training_instances


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-data', help='Data to train the model',
                        dest='input_file', required=True)
    parser.add_argument('-dim', help='Vector dimensionality',
                        dest='vec_dim', default=50, type=int)
    parser.add_argument('-eta', help='Learning rate', dest='eta',
                        default=0.025, type=float)
    parser.add_argument('-win', help='Size of the context window',
                        dest='win_size', default=2, type=int)
    parser.add_argument('-epochs', help='Number of passes over training data',
                        dest='epochs', default=1, type=int)
    parser.add_argument('-min_occ', help='''Minimum number of times a word must
                                          appear in the corpus to be
                                          included in the vocabulary''',
                        dest='min_occ', default=5, type=int)
    parser.add_argument('-sent_len', help='Minimum sentence length',
                        dest='sent_len', default=5, type=int)
    args = parser.parse_args()

    print("Reading data...")
    data = utils.read_data(args.input_file)
    vocab = utils.vocabulary()
    sentences = utils.normalize(data, min_word_occurences=args.min_occ,
                                min_sentence_length=args.sent_len)
    sentence_list = list(utils.docs2bow(sentences, vocab))
    rng = np.random.RandomState(23)
    print("Data loaded!")
    print("{0} words in the vocabulary.".format(len(vocab)))
    print("{0} sentences in the training data.".format(len(sentences)))
    print("Starting training.")
    model = CBOW(vocab, rng, hidden_size=args.vec_dim, learning_rate=args.eta)
    train(model, sentence_list, args.win_size, args.epochs)
