# Feature Selection

All feature selection methods have two logical components. The first component is the chosen search technique for subsets of a given set of features.
The second component is the chosen evaluation measure to score the different feature subsets.
There are many methods in this space and they all differ in the evaluation measure they use.

We can classify the methods into three classes:

1. The **wrapper methods** train a new model for each feature subset. They effectively wrap around the core task of training a model, which is performed by any learner. The score is calculated by counting the number of mistakes on the hold-out set.
This method is expensive yet usually provides the best performing feature set. An important wrapper method is forward stepwise selection
2. The **filter methods** use a proxy measure as a score instead of the actual test error. They are used to expose relationships between features. The typical scores which are used for this method class are mutual information and Pearson correlation coefficient.
These methods are fast to compute and the resulting feature set is not tuned to a specific type of models.
3. The **embedded methods** perform feature selection while constructing the model. The LASSO regularisation. The elastic net regularisation combines the l1 and l2 regularisations.

## Forward Stepwise Selection

The forward stepwise selection is a wrapper method. It starts with an empty set of features and adds one feature at a time. The feature which improves the model performance the most is added to the feature set. This process is repeated until no further improvement is observed.
It is common in practice to want to add another feature to a model with k existing features, with the goal of lowering the generalisation error. We can therefore try out the new model over the extended set of features.
It is much cheaper computationally than going through all possible subsets of the set of n features, since the number of such subsets of exponential in n (to be precise: $2^n$).
The forward stepwise selection algorithm only computes $O(n^2)$ models: At step i, it trains and tests n − i + 1 new models, therefore, taking a computational complexity of $O(n^2)$ models to train and test in total.
A special case is that of selection for linear regression models: The computational complexity can be reduced to that of training only one model [Efroymson, 1960]. An open-source implementation of this approach is available on [GitHub](https://www.github.com/EFavDB/linselect).

## Feature Selection via Correlation Coefficient

The (Pearson) correlation coefficient normalises the covariance to give a value between −1 and +1:
$corr(X, Y ) = \frac {cov(X,Y)} {\sqrt {cov(X, X) * cov(Y, Y)}}$

If a correlation value is equal to 0, it does not necessarily mean that the variables are independent.
**Correlation of 0 is a necessary condition for independence, not a sufficient one.**

## Mutual Information

Given two random variables X and Y , the mutual information I(X, Y ) quantifies how much X discloses about Y. Formally:

$I(X,Y) = \sum_x \sum_y p(X=x,Y=y)*log(\frac {p(X =x,Y =y)} {p(X =x)·p(Y =y)}) $

The above formula for mutual information assumes that the variables have a finite discrete domain.
In case of numerical domains, we first have to bucketize them, that is, create a histogram that has the desired number of buckets, which defines the size of the domain of the variable with one bucket representing one value in the domain.
The mutual information is a symmetric measure, that is, $I(X, Y ) = I(Y, X)$.
