# Second Order optimisation

[TOC](https://github.com/Abby3017/machine_learning_refined/blob/main/sample_chapters/2nd_ed/chapter_4.pdf)

## Second-Order Optimality Conditions

Condition of convexity: $\frac {\partial^2}{\partial w^2} g(v) \le 0$ or $\ge 0$.<br/>
Function g with multi-dimensioal input: hessian matrix evaluated at a point v, denoted by $\nabla^2 g(v)$ should have all non-negative or non-positive eigenvalues.
Then g is said to be convex (or concave) at v, Hessian Matrix itself is called positive (or negative) semi-definite.

A function $g(w)$ is said to be convex everywhere if its second derivative is always non-negative. Likewise $g(w)$ is convex everywhere if its second derivative is always non-negative eigenvalues. **Second order definiton of convexity**.

## Geometry of Second-Order Taylor Series

Second order Tayler series approximation of a function g at point v is:
$h(w) = g(v) + (\frac {\partial} {\partial w} g(v))(w-v) + \frac{1}{2}(\frac {\partial^2}{\partial w^2}g(v)) (w-v)^2$
</br>
Not only does the **second-order approximation match the curvature of the underlying function at each point v** in the function’s domain, but if the function
is convex at that point (due to its second derivative being nonnegative) then the second-order Taylor series is convex everywhere.
Likewise if the function is concave at v, this approximating quadratic is concave everywhere.

## Newton's Method

Gradient of the quadratic approximation to zero gives stationary point:
$w^* = v - (\nabla^2 g(v))^{-1} \nabla g(v)$
</br>
At the kth step of this process for a single-input function, second order Taylor approximation centered at point $w^{k-1}$ is:
$h(w) = g(w^{k-1}) + g'(w^{k-1})(w-w^{k-1}) + \frac{1}{2}g''(w^{k-1})(w-w^{k-1})^2$

updated $w^k$ as: $w^k = w^{k-1} - \frac{g'(w^{k-1})}{g''(w^{k-1})}$
Here, we invert the Hessian matrix at each step, which is computationally expensive.

This reliance on quadratic information also makes Newton’s method naturally more dificult to use with nonconvex functions since at concave portions of such a function the algorithm can climb to a local maximum.

**Newton's method was created as zero finding algorithm where f(w) = 0.**

``` python
# import autograd’s automatic differentiator
from autograd import grad
from autograd import hessian
# import NumPy library
import numpy as np

# Newton’s method
def newtons_method(g, max_its, w, **kwargs):
# compute gradient/Hessian using autograd
    gradient = grad(g)
    hess = hessian(g)
    # set numerical stability parameter
    epsilon = 10**(-7)
    if ’epsilon’ in kwargs:
    epsilon = kwargs[’epsilon’]

    # run the Newton’s method loop
    weight_history = [w] # container for weight history
    cost_history = [g(w)] # container for cost function history
    for k in range(max_its):
        # evaluate the gradient and hessian
        grad_eval = gradient(w)
        hess_eval = hess(w)
        # reshape hessian to square matrix
        hess_eval.shape = (int((np.size(hess_eval))**(0.5)),int((np.size(hess_eval))**(0.5)))
        # solve second-order system for weight update
        A = hess_eval + epsilon*np.eye(w.size)
        b = grad_eval
        w = np.linalg.solve(A, np.dot(A,w)-b)
        # record weight and cost
        weight_history.append(w)
        cost_history.append(g(w))
    
    return weight_history,cost_history
```

Two limitation of Newton's method:

- The first is that it is computationally expensive to compute the Hessian matrix at each step. It depends on the dimension of the input. For this, replace hessian with approximation thats doesn't suffer from scaling
- The second is that the Hessian matrix must be invertible at each step. This is not always the case. To avoid that, we add small value to make it invertible
- It doesn't work well with non-convex functions, for that another approach needs to be specified.
