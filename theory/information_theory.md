# Information Theory

This document will consists theory of information theory related to machine learning.

## Information Bottleneck (IB) theory

In the Information Bottleneck (IB) theory [Tishby et al., 1999](https://arxiv.org/abs/physics/0004057), given data (X, Y), the optimization target is to minimize mutual information I(X; T) while T is a sufficient statistic satisfying
$I(T ; Y) = I(X; Y)$

Here T is set of features from X that are relevant to Y.

[Shwartz-Ziv & Tishby](https://arxiv.org/abs/1703.00810) designed an experiment about DNN and found that the intermediate feature T of DNN meets the IB theory: maximize I(T ; Y ) while minimizing I(X; T).
