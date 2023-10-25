# Gradient Descent

## Normalised Gradient Descent

If a function has large flat regions with stationary points, magnitude of gradient in such regions would be quite small. As gradient would crawl in these areas, due to saddle point it will vanishes. As in vanishing gradient magnitude is getting smaller we can ameliorate this problem by removing magnitude.
That can be done by normalising gradient.

Length of standard gradient descent step: $ \alpha || \Delta g(w^{k-1}) ||_2  $ <br/>
Normalised gradient descent step form, to ignore magnitude: $w^k = w^{k-1} - \alpha \frac {\Delta g(w^{k-1})} {|| \Delta g(w^{k-1}) ||_2}$

After normalising gradient descent, length of each step is exactly $\alpha$.
This can be interpreted differently as learning rate adjust itself based on magnitude. Constant is added to avoid zero division.</br>

$ \frac{\alpha}{|| \Delta g(w^{k-1}) ||_2 + \epsilon} $

[Normalising Gradient Descent](https://jermwatt.github.io/machine_learning_refined/notes/3_First_order_methods/3_9_Normalized.html)

## Normalising out the magnitude component wise

As gradient is vector of N partial derivatives, $j^{th}$ partial derivative shows how the gradient behave along the $j^{th}$ direction. </br>
Normalising the $j^{th}$ partial derivative by full magnitude, $\frac {\frac{\partial}{\partial w_j} g(w)} {\sqrt \sigma_{n=1}^N (\frac { \partial }{\partial w_n} g(w))^2}$
$j^th$ gradient is normalised using sum of magnitude of all partial derivatives. if the $j^th$ gradient is too small, it will erase virtually all of its contribution to the final gradient step.
This can be issue in flat region of function alongside some of weights/partial derivative, as it diminishes the contribution of gradient in that direction.
In this case, we can normalise the magnitude component-wise. $sign(\frac{\partial}{\partial w_j} g(w))$. (do calculation, it will give same result)<br/>
Component wise normalise step is : $w^k = w^{k-1} - \alpha sign(\Delta g(w^{k-1}))$

If we normalise gradient component wise, then length of component normalised gradient step is: $ \sqrt N \alpha$

[Normalising Gradient Descent](https://jermwatt.github.io/machine_learning_refined/notes/3_First_order_methods/3_9_Normalized.html)

Another Blog on [Normalising Gradient Descent](https://rezaborhani.github.io/mlr/blog_posts/Mathematical_Optimization/Part_2_gradient_descent.html)

## Usage

in normalizing every step of standard gradient descent we do shorten the first few steps of the run that are typically large (since random initializations are often far from stationary points of a function).
This is the trade-off of the normalized step when compared with the standard gradient descent scheme: we trade shorter initial steps for longer ones around stationary points.
Normalized step described  is normalized with respect to the  l2 norm we will see here that the component-normalized step is similarly normalized with respect to the  $l\infin$ vector norm
