# Maximum Entropy Model

Maximum entropy classification is a method that generalizes **logistic regression to multiclass problems**. The Maximum Entropy model is a type of log-linear model.
It is a classical machine learning approach.

If we are given some data and told to decide, we could think of attributes about the data,i.e.,  features. Some of these features might be more important than others. We apply a weight to each feature found in the data, and we add up all of the features.
Finally, the weighted sum is normalized to give a fraction between 0 and 1. We can use this fraction to tell us the score of how confident we might be in making a decision.

According to ME principle, a model with good generalization capability is expected to have as less extra hypothesis on data (X, Y ) as possible.
Softmax regression assumes independent observations when applying ME principle

This document explains how max ent models is derived from the conditional probability [distribution](https://web.stanford.edu/class/archive/cs/cs224n/cs224n.1162/handouts/MaxentTutorial-16x9-MEMMs-Smoothing.pdf).

(Original Maximum Entropy Model) Supposing the dataset has input X and label Y , the task is to find a good prediction of Y using X. The prediction $\hat{Y}$ needs to maximize the conditional entropy $H(\hat{Y}|X)$ while preserving the same distribution with data (X, Y ).
$$
\begin{align*}
\text{min} -H(\hat{Y}|X) \\
s.t \quad P(\hat{Y}|X) = P(Y|X)
\end{align*}
$$

## Log Linear Model

The log-linear model uses a linear combination of features and weights to find the predicted label with maximum log-likelihood.

$P(y|x;w) = \frac{\exp^{\Sigma_j w_j f_j(x,y)}} {\Sigma_{y`} \exp^{\Sigma_j w_j f_j(x,y`)}}$

Function $f(x,y)$ represent relation between data and label. $w_j$ is the weight of feature $f_j$ which captures the importance of the feature.

## Approach

The maximum entropy model is a discriminative model, which means that it models $P(y|x)$ directly.
In the training phase, we have to find the weights $w_j$ that maximize the log-likelihood of the training data.

$L(w) = \Sigma^n_i log(p_i| x_i;w)$

L(w) measures how well w explains the training data. The goal is to find the weights w that maximize L(w).

$ \tilde{w} = argmax_w L(w)$

The process involves iterating through training data many iterations:

- Initially, initialize the w to some random values.
- Keep iterating through each input. During each iteration, we update the weight by finding the derivative of L(w) concerning wj.
- Updating vector was below and repeated until converged.

## Maximum Entropy Likehood

The maximum entropy principle states that we have to model the given set of data by finding the highest entropy to satisfy the constraints of our previous knowledge.

## Maximum Entropy Markov Model

There are many systems where there is a time or state dependency. These systems evolve through a sequence of states, and past states influence the current state. For example, stock prices, DNA sequencing, human speech, or words in a sentence.

Maximum Entropy Markov Model makes use of state-time dependencies,i.e., it uses predictions of the past and the current observation to make the current prediction.

The MEMM has dependencies between each state and the full observation sequence explicitly. MEMM has only one transition probability matrix. This matrix encapsulates previous states y(i−1) and current observation x(i) pairs in the training data to the current state y(i).
Our goal is to find the P(y1,y2,…,yn|x1,x2,…xn). This is given by:

$P(y_1 ... y_n) = \Pi_{i=1}^n (P(y_n| y_1...y_{i-1}, x_1 ... x_n))$

Since HMM only depends on the previous state, we can limit the condition of y(n) given y(n-1). This is the Markov independence assumption.
$P(y_1 ... y_n) = \Pi_{i=1}^n (P(y_n| y_{i-1}, x_1 ... x_n))$

So MEMM defines using Log-linear model as:

$$P(y_i|y_{i-1}, x) = \frac{\exp^{\Sigma_{i=1}^n w_i f_i(y_i, y_{i-1},x)}} {\Sigma_{y`} \exp^{\Sigma_{j=1}^{i-1} w_j f_j(x,y_j)}}$$

## Interview Questions

Q1. What is the condition for maximum entropy?

Ans. The principle of maximum entropy states that the probability distribution that best represents the current state of knowledge about a system is the one with the most significant entropy in the context of precisely stated primary data.

Q2. Is maximum entropy possible?

Ans. The maximum entropy principle (MaxEnt) states that the most appropriate distribution to model a given set of data is the one with the highest entropy among all those that satisfy our prior knowledge's constraints.

Q3. Which distribution has maximum entropy?

Ans. Therefore, the normal distribution is the maximum entropy distribution with a known mean and variance.

[Coding Ninja - basic](https://www.codingninjas.com/studio/library/maximum-entropy-model)

## Logistic Regression vs Maximum Entropy

In Max Entropy the feature is represented with f(x,y), it mean we can design feature by using the label y and the observerable feature x, while, if f(x,y) = x it is the situation in logistic regression.

## NLP examples

In NLP task like POS, it is common to design feature's combining labels. For example: current word ends with "ous" and next word is noun. it can be feature to predict whether the current word is adjective.

## UNDERSTANDING DEEP LEARNING GENERALIZATION BY MAXIMUM [ENTROPY](https://arxiv.org/pdf/1711.07758.pdf)

This paper attempts to explain generalisation of DNN using maximum entropy. DNN is then regarded as approximating the feature conditions with multilayer feature learning, and proved to be a recursive solution towards maximum entropy principle.
It also shows how shortcuts and regularization improves model generalization explained by maximum entropy.

Models fulfilling the principle of ME make least hypothesis beyond the stated prior data, and thus lead to least biased estimate possible on the given information.
Different selections of feature functions lead to different instantiations of maximum entropy models. The most simple and wellknown instantiation is that ME principle invents identical formulation of softmax regression by selecting certain feature functions and treating data as conditionally independent.
