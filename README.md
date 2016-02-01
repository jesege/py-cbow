# py-cbow

A very simple implementation of the continuous-bag-of-words model from word2vec [1].
Neither hierarchical softmax or negative sampling is used during training, meaning
that it does not scale to large vocabularies. It is not intended as an actual usable module,
but was written purely as an exercise.

The code is based on the explanations in [2] and with some inspiration from [3].


## References

[1] [Tomas Mikolov, Kai Chen, Greg Corrado, and Jeffrey Dean. Efficient estimation of word representations in vector space. *arXiv preprint arXiv:1301.3781*, 2013](http://arxiv.org/abs/1301.3781)

[2] [Xin Rong. word2vec parameter learning explained. *arXiv preprint arXiv:1411.2738*, 2014](http://arxiv.org/abs/1411.2738)

[3] [http://www.folgertkarsdorp.nl/word2vec-an-introduction](http://www.folgertkarsdorp.nl/word2vec-an-introduction)