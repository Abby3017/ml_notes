# [Word2Vec](https://arxiv.org/abs/1301.3781)

The first was called “Continuous Bag of Words” where need to predict the center words given the neighbor words.
<img src="https://amitness.com/images/nlp-ssl-center-word-prediction.gif" alt="cbow"/>

The second task was called “Skip-gram” where we need to predict the neighbor words given a center word.
<img src="https://amitness.com/images/nlp-ssl-neighbor-word-prediction.gif" alt="skip"/>

The main difference between these two is that CBOW is faster while skip-gram is slower but does a better job for infrequent words.
Representations learned had interesting properties such as this popular example where arithmetic operations on word vectors seemed to retain meaning.

## Limitations of Word2Vec

Word2Vec is a great technique to learn word embeddings but it has its own limitations. Some of them are:

- It doesn’t capture polysemy (multiple meanings of a word).
- It doesn’t capture multiple senses of a word.
- It doesn’t capture out of vocabulary words.
- It doesn’t capture relationships between words which are not co-occurring.
