# First Order Optimisation

Gradient is first derivative of a function. Optimisation algorithm that leverage first derivative is collectively called first order optimisation.

## First Order Optimality Condition

First order optimality condition is a condition that must be satisfied by a local minimum of a function. It is a necessary condition for a local minimum.
Tangent to the function at a local minimum is also known as the first order Taylor approximation of the function at that point. The first order optimality condition is that the first order Taylor approximation is greater than or equal to the function value at that point.

**Key Point** -- Minimum values of a function are naturally located at valley floors where a tangent line or hyperplane is flat, thus has zeroed value slope or gradient.

First order sytem equation, $ \Delta g(v) = 0_{N*1} $
[Machine Learning Refined Chapter 3](https://github.com/jermwatt/machine_learning_refined/blob/main/sample_chapters/2nd_ed/chapter_3.pdf)

## Coordinate descent

Coordinate descent is an optimisation algorithm that successively minimises along coordinate directions to find the minimum of a function. Coordinate descent is a first order optimisation algorithm. This is effective if each of these equations can be solved in closed form.
To solve the first order equation, we first initialize the vector $w^0$ and begin by updating first coordinate by solving:
        $\frac {\partial} {\partial w_1} g(w^0) = 0$
for the optimal first weight $w_1^*$. Then we update the second coordinate by solving:
        $\frac {\partial} {\partial w_2} g(w_1^*) = 0$
In solving for this equations for $w_1$ weights all other weights are kept fixed at the initial values.
For example, $g(w) = w_1^2 + w_2^2 + 2w_1w_2$, initial weights $w^0 = [1, 1]^T$. <br/>
        Step 1: $\frac {\partial} {\partial w_1} g(w^0) = 2w_1 + 2w_2 = 0$ <br/>
        Step 2: $\frac {\partial} {\partial w_2} g(w_1^*) = 2w_2 + 2w_1^* = 0$ <br/>
        Step 3: $w_1^* = -w_2$ and $w_2^* = -w_1^* = w_2$ <br/>
        Step 4: $w^1 = [-1, -1]^T$ <br/>

As we can see in step 2, updated weight of step 1 i.e $w_1^*$ is used to update weight of $w_2$. This whole step helps to reach new weight $w^1$.
Now this step is repeated until convergence. $w_2$ is reached through $w_1$. [Machine Learning Refined Chapter 3](https://github.com/jermwatt/machine_learning_refined/blob/main/sample_chapters/2nd_ed/chapter_3.pdf)

## Concept of Gradient Descent

[Machine Learning Refined Chapter 3](https://github.com/jermwatt/machine_learning_refined/blob/main/sample_chapters/2nd_ed/chapter_3.pdf)

**Negative gradient is always points perpendicular to the contours of the function.**

## Weakness of Gradient Descent

Gradient descent crawl slowly near minima and saddle point.
Zig-zagging issue due to oscillation.
