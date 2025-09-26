# Mathematics for Machine Learning

## Vectors

An object which can be added to other objects of the same type and multiplied by scalars resulting in another object of the same type is called a vector. There are fundamental different kind of vectors:

• Geometric vectors: arrows in space with a direction and a magnitude. Least abstract vector.
• Algebraic vectors $\Reals^n$: ordered tuples of numbers, which can be added and multiplied by scalars. Most abstract vector.
• Functions/Polynomials: can be added and multiplied by scalars. Polynomials are vector subspace of space of functions.

The sum of two vectors lies in the same plan as the two vectors.

The sum and multiplication operations of vectors should satisfy the following properties:
• Commutativity: u + v = v + u
• Associativity: u + (v + w) = (u + v) + w
• Distributivity: a(u + v) = au + av

Commutativity makes sure the order of addition does not matter.
Associativity makes sure the order of operations does not matter and we can add multiple vectors. *Matrix operation multiplication satisfy associativity but not commutativity*.

**Linear algebra is all about satisfying these properties of vectors and its operations.**

Addition of vectors involves joining vectors from head to tail and then drawing a vector from the tail of the first vector to the head of the last vector. For subtraction, we reverse the direction of the vector to be subtracted and draw from the tip of the subtracted vector to the tip of the first vector. This is tip to tail geometric method.
Another way to remember (algebraic way) it points in such a way if you add $b$ to it you will get $a$. For example, view the [figure at](../img/vector_add_subtract_geometric_explain.pdf).

*Polynomials* are subset of functions. They follow the properties of vectors. Its defined for example the quadratic polynomials following the condition upto degree 2 so even linear or constant polynomials are also part of quadratic polynomials.
Two polynomials of same degree can be added to get a polynomial in same subspace. By this property resultant polynomial will not able to get out of the subspace. Even polynomials like having root at given point have subspace with other polynomials of same properties. Any polynomial in that subspace won't be able to get out of the subspace by addition or scalar multiplication.

*Sets of n number* is tuple of n-set of numbers. For example, (1,2,3) is a 3-set of numbers.

Linear combinations is the sum of scalar multiples of vectors. For example, given vectors $v_1, v_2, v_3$ and scalars $a_1, a_2, a_3$, the linear combination is given by $a_1v_1 + a_2v_2 + a_3v_3$.

### Span

The span of a set of vectors is the set of all possible linear combinations of those vectors. For example, given vectors $v_1, v_2, v_3$, the span is given by $\{a_1v_1 + a_2v_2 + a_3v_3 | a_1, a_2, a_3 \in \Reals\}$. Its a geometric concept.
**All possible linear combinations of a set of vectors form a subspace called the span of those vectors.**

In geometry, the span of two vectors is a plane if they are not collinear. The span of three vectors is the entire space if they are not coplanar.

The span concept of geometric vectors can be extended to  $R^n$ vectors and polynomials. However, based on the properties of the vectors, the span may not cover the entire space.

**Spanning sets**: A set of vectors is said to span a space if their span is equal to the entire space of vectors. For 2-dimensional space, two non-collinear vectors are sufficient to span the entire space. For 3-dimensional space, three non-coplanar vectors are sufficient to span the entire space.

For a **linear subspace**, it should contain the zero vector, be closed under addition and scalar multiplication. If a vector multiplied by zero gives zero vector, then the zero vector should be part of the subspace.
In linear subspace, addition of two vectors should give a vector in the same subspace. Similarly, scalar multiplication of a vector should give a vector in the same subspace.

Linearly dependent vectors

Span of vectors

Vector norms

## Matrices

Invertible matrices

Several equivalent characterisations exist to determine if a square matrix A is invertible or not. Depending on the use case, one might be preferred over the other.

• A is invertible if and only if the columns of A are linearly independent.
• A is invertible if and only if the rows of A are linearly independent.
• A is invertible if and only if the determinant of A is non-zero: det(A) = |A| ≠ 0.
