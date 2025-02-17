# Probability

A probabilistic approach to Machine Learning aims at **modelling the noise in the data explicitly**. It picks a model for the data and formulates the deviation (or uncertainty) of the data from the model. It uses notions of probability to define and measure the suitability of various models.
It then “finds” model parameters or makes predictions on unseen data using these suitability criteria.

## Joint Probability

$P(A, B) = P(A|B)P(B)$
joint probability of A and B is the probability of B times the probability of A given B.
A joint model gives probabilities P(A,B) and tries to maximize this joint likelihood. Just by counting and getting their proability get us the weight.
Relative frequencies give maximum joint likelihood on categorical data.

if A and B are independent events, then $P(A,B) = P(A) P(B)$

If we want to reason about more than one random variable, then we consider joint probability distributions for vectors of random variables, e.g., p(X = x, Y = y).

## Conditional Probability

$P(A|B) = \frac {P(A, B)}{P(B)}$

The intuition of the above formulas is as follows: If we know that event B has already occurred and we want to know the probability of A under this premise, then we compute the probability that A and B occur relative to the probability that B occurs.
In other words, we can exclude the possibility that B does not occur.
Computing p(A|B) comes down to calculating the fraction of the intersection of A and B out of B.
It is harder to do and more closely related to classification error.

### Factorization

$P(A| B,C) \varpropto P(A|B) * P(A|C)$

Given any two nonzero real numbers 𝑥 and 𝑦 , it is always the case that $𝑥 \varpropto 𝑦$ : just choose the constant of proportionality to be $𝑥𝑦$. For independent 𝐵 and 𝐶 , it is always the case that
$𝑃(𝐴𝐵𝐶)\varpropto𝑃(𝐴𝐵)𝑃(𝐴𝐶)$ and hence $𝑃(𝐴∣𝐵,𝐶)\varpropto𝑃(𝐴∣𝐵)𝑃(𝐴∣𝐶)$.

$P(A,B|C) = P(A|B,C)P(B|C)$ if A and B are independent to each other then $P(A,B|C) = P(A|C)P(B|C)$

## Chain rule of Probability

chain rule of two random events A and B is:
$P(A, B) = P(A|B)P(B) = P(B|A)P(A)$

The chain rule for conditional probability writes the joint distribution of n random variables X(1), . . . , X(n) as a product of conditional probabilities: <br/>
$p(X^{(1)},...,X^{(n)})=p(X^{(1)})·\prod^n_{i=2}(X^{(i)} |X^{(1)},...,X^{(i−1)})$
<br/>
$= p(X^{(1)}) · p(X^{(2)}|X^{(1)}) · p(X^{(3)}|X^{(1)}, X^{(2)}) · · · p(X^{(n)}|X^{(1)}, . . . , X^{(n−1)})$

## Law of Total Probability

Given a partition B1,...,Bn of S with p(Bi) > 0 we can write the probability of an event A as

$P(A) = \sum _i P(A|B_i)P(B_i)$

## Marginal Probability

$P(A) = \sum _i P(A, B_i)$

probability of A is the sum of the joint probabilities of A and all other events B.

## Posterior Probability

posterior probability = $\frac {conditional probability  *  prior probability}{ evidence }$

## Bayes Theorem

$P(A|B) = \frac {p(B|A)  p(A)}{p(B)}$

It states, for two events A & B, if we know the conditional probability of B given A and the probability of B, then it’s possible to calculate the probability of A given B.

## Random Variable

A random variable is a variable whose possible values are numerical outcomes of a random phenomenon. There are two types of random variables, discrete and continuous.
A random variable X is a function from a sample space to some numeric domain (usually R).
If x is a value from the domain of X, then p(X = x) denotes the probability of the event ${s ∈ S : X(s) = x}$. We write X ∼ p(x) (read as “X follows probability distribution p(x)”) to specify the probability distribution of the random variable X.

A random variable X is called a discrete random variable if there are countably many a1, a2, . . . such that $\sum_{aj} p(X = a_j ) = 1$.
The distribution of X is then given by the probability mass function (PMF) pX , where $pX (x) = p(X = x)$.
The cumulative distribution function (CDF) maps x to p(X ≤ x). It is called cumulative, as it accumulates the probabilities as we move along the domain of X.

A random variable X is a continuous random variable if it’s CDF is continuous. In this case, the domain of X is infinite.
Therefore, we integrate over all x, instead of summing over them, to get a probability of 1. The probability of a particular value x would be 0; for that reason we consider intervals in the domain of x.
$\intop ^\infty_{_\infty} p(x)dx=1 $ <br/>
$P(a≤ X ≤b)= \intop ^b _a p(x)dx$ <br/>
The probability density function (PDF) p(x) is the derivative of the CDF. The PDF is the distribution of X.

## Expectation & Variance

The expected value of a random variable X is denoted by E[X]. If X is a:

- discrete random variable: $E[X] = \sum_{x\in dom(X)} x . p(x)$
- continuous random variable: $E[X] = \int x · p(x) dx$ <br/>
The expectation can be thought of as the average outcome or the mean. It is the sum of all possible outcomes, weighted by their probabilities. An important property is the linearity of expectation:
$E[\alpha · X + \beta · Y ] = \alpha · E[X] + \beta · E[Y]$

for constants α and β.

The variance of a random variable measures how much the values of a probability distribution vary around the expectation on average if randomly drawn:

$Var(X) = E[(X − E[X])2] = E[X2] − E[X]2$<br/>

Some properties of the variance are given below:

- $Var(\alpha·X+\beta)=α2·Var(X)$ for constants $\alpha$ and $\beta$
- If X and Y are independent, then the variance of the sum is given by the sum of the variances: Var(X + Y ) = Var(X) + Var(Y )

## Standard Deviation and Covariance

The standard deviation of a random variable X is the square root of the variance of X:
$ SD(X) = \sqrt {Var(X)}$

The covariance generalises variance to two random variables. It is a measure of how a random variable X behaves in relation to another random variable Y . The covariance is defined as:
$Cov(X,Y)=E[(X−E[X])(Y −E[Y])]=E[XY]−E[X]E[Y]$

The covariance of a random variable with itself is equal to its variance: $Cov(X,X) = Var(X)$.
Covariance is symmetric, i.e., $Cov(X, Y ) = Cov(Y, X)$.

## Maximum Likelihood Estimation

Assume we are given observations $x_1,...,x_N$ drawn independently from the same distribution p. Such observations are called independently and identically distributed (in short, i.i.d.).
We assume p to have a parametric form, where vector $\theta$ contains all parameters relevant to p. For example, if p is the probability mass function of a Bernoulli random variable, $\theta$ consists of the probability of success.
In case p is the probability density function of a Gaussian random variable, then $\theta$ consists of the mean and variance of the Gaussian Distribution.

The likelihood of observing $x_1, . . . , x_N$ is defined as the probability of making these observations assuming that they are generated according to the distribution p.
We denote the joint probability distribution of the observations given θ as $p(x_1, . . . , x_N |θ)$.

**Maximum Likelihood Principle: Pick parameter $\theta$ that maximises the likelihood.**

The idea of the maximum likelihood principle is to find a parameter vector θ∗ that
maximises the likelihood:<br/>
$\theta_∗ = argmaxp(x_1,...,x_N | \theta)$ <br/>

Such a parameter vector is called maximum likelihood estimator (MLE) and usually denoted as $\theta_{ML}$.
Because of the i.i.d. assumption, we can factorise the likelihood of observing $x_1,...,x_N$ into the product of the likelihoods of the individual observations: <br/>
$p(x_1,...,x_N | \theta) = \prod^N_{i=1} p(x_i| \theta)$ <br/>

## Various Distribution

### Gaussian Distribution

In the univariate case, the density function p of the normal distribution is given by:
$p(x|μ,σ^2)=\frac{1}{\sqrt{2πσ^2}}e^{−\frac{(x−μ)^2}{2σ^2}}$

where μ is the mean and $ \sigma^2$ is the variance of the distribution.

### Laplace Distribution

density function for the Laplace distribution is given by:
<br/>
$p(x|μ,b)=\frac{1}{2b}e^{−\frac{|x−μ|}{b}}$
<br/>
where μ is the mean and b is the scale parameter.

### Exponential Distribution

The exponential distribution is a special case of the gamma distribution, where the shape parameter is equal to 1. The density function of the exponential distribution is given by:
<br/>
$p(x|λ)=λe^{−λx}, for \, x \ge 0$
<br/>
where λ is the rate parameter.

### Poisson Distribution

The Poisson distribution is a discrete probability distribution that expresses the probability of a given number of events occurring in a fixed interval of time or space if these events occur with a known constant mean rate λ and independently of the time since the last event.
<br/>
$p(x|λ)=\frac{λ^x}{x!}e^{−λ}$
<br/>
where λ is the rate parameter.

### Dirichlet Distribution

The Dirichlet distribution is a multivariate generalisation of the beta distribution. It is a distribution over vectors of positive real numbers that sum to 1. The density function of the Dirichlet distribution is given by:
<br/>
$p(x|α)=\frac{1}{B(α)}\prod^k_{i=1}x^{α_i−1}_i$
<br/>
where α is a vector of positive real numbers and B(α) is the multivariate beta function.

### Multinomial Distribution

The multinomial distribution is a generalisation of the binomial distribution. It is a distribution over vectors of non-negative integers that sum to n. The density function of the multinomial distribution is given by:
<br/>
$p(x|n,π)=\frac{n!}{\prod^k_{i=1}x_i!}\prod^k_{i=1}π^{x_i}_i$
<br/>
where n is a positive integer and π is a vector of probabilities that sum to 1.

### Beta Distribution

The beta distribution is a distribution over the interval [0, 1]. It is a conjugate prior for the binomial distribution. The density function of the beta distribution is given by:
<br/>
$p(x|α,β)=\frac{1}{B(α,β)}x^{α−1}(1−x)^{β−1}$
<br/>
where α and β are positive real numbers and B(α, β) is the beta function.

### Gamma Distribution

The gamma distribution is a distribution over the interval [0, ∞). It is a conjugate prior for the exponential distribution. The density function of the gamma distribution is given by:
<br/>
$p(x|α,β)=\frac{β^α}{Γ(α)}x^{α−1}e^{−βx}$
<br/>
where α and β are positive real numbers and Γ(α) is the gamma function.
