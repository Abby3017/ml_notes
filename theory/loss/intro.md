# Introduction

## Ranking loss

Objective of ranking loss is to predict relative distances between inputs. They need similarity scores between pairs of inputs.
We train the feature extractors to produce similar representations for both inputs, in case the inputs are similar, or distant representations for the two inputs, in case they are dissimilar.

### Pairwise Ranking Loss

In this setup positive and negative pairs of training data points are used. Positive pairs are composed by an anchor sample $x_a$ and a positive sample $x_p$, which is similar to $x_a$ in the metric we aim to learn, and negative pairs composed by an anchor sample $x_a$ and a negative sample $x_n$.
