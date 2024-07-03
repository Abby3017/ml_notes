# Introduction to Contrastive Learning

Contrastive learning is a type of self-supervised learning that aims to learn representations by maximizing the similarity between similar samples and minimizing the similarity between dissimilar samples.
It is a popular method for learning representations from unlabeled data, and has been shown to be effective in a wide range of domains, including computer vision, natural language processing, and speech recognition.

## Theoretical Explanation

Two main metrics for evaluating embeddings as defined by [Wang, Isolar](https://arxiv.org/abs/2005.10242) are:

1. **alignment**: alignment(closeness) of features from positive pairs. It measures noise-invariance property.
2. **uniformity**: uniformity of the induced distribution of the features on the hypersphere. How uniformly distributed are the features on the hypersphere.

## References

[ETH contrastive presentation](https://disco.ethz.ch/courses/fs22/seminar/talks/03_05_contrastive_presentation%20.pdf)
[Inductive Biases Contrastive Learning](https://proceedings.mlr.press/v162/saunshi22a/saunshi22a.pdf)
[Bound for loss](http://www.offconvex.org/2019/03/19/CURL/)
