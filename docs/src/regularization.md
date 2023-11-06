# Regularization
When formulating inverse problems, a Regularizer is formulated as an additional term in a cost function, which has to be minimized. Popular optimizers often deal with a regularizers $g$, by computing the proximal map

```math
\begin{equation}
  prox_g (\mathbf{x}) = \underset{\mathbf{u}}{argmin} \frac{1}{2}\vert\vert \mathbf{u}-\mathbf{x} \vert {\vert}^2 + g(\mathbf{x}).
\end{equation}
```

In order to implement those kinds of algorithms,RegularizedLeastSquares defines the following type hierarchy:
```julia
abstract type AbstractRegularization
prox!(reg::AbstractRegularization, x)
norm(reg::AbstractRegularization, x)
```
Here `prox!(reg, x)` is an in-place function which computes the proximal map on the input-vector `x`. The function `norm` computes the value of the corresponding term in the inverse problem. RegularizedLeastSquares.jl provides `AbstractParameterizedRegularization` and `AbstractProjectionRegularization` as core regularization types.

## Parameterized Regularization Terms
This group of regularization terms features a regularization parameter `λ` that is used during the `prox!` and `norm `computations. Examples of this regulariztion group are `L1`, `L2` or `LLR` (locally low rank) regularization terms.

These terms are constructed by supplying a `λ` and optionally term specific keyword arguments:

```julia
l2 = L2Regularization(0.3)
```
Parameterized regularization terms implement:
```julia
prox!(reg::AbstractParameterizedRegularization, x, λ)
norm(reg::AbstractParameterizedRegularization, x, λ)
```
where `λ` by default is filled with the value used during construction.
Invoking `λ` on a parameterized term retrieves its regularization parameter. This can be used in a solver to scale and overwrite the parameter as follows:
```julia
prox!(l2, x, λ(l2)*0.2)
```

## Projection Regularization Terms
This group of regularization terms implement projections, such as a positivity constraint or a projection with a given convex projection function.

```julia
positive = PositiveRegularization()
prox!(positive, [2.0, -0.2]) == [2.0, 0.0]
```

## Nested Regularization Terms
Nested regularization terms are terms that act as decorators to the core regularization terms. These terms can be nested around other terms and add functionality to a regularization term, such as scaling `λ` based on the provided system matrix or applying a transform, such as the Wavelet, to `x`:
```julia
core = L1Regularization(0.8)
wop = WaveletOp(Float32, shape = (32,32))
reg = TransformedRegularization(core, wop)
```