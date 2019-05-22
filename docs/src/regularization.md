# Regularization

## Defining a Regularizer
When formulating inverse problems, a Regularizer is formulated as an additional term in a cost function, which has to be minimized. Popular optimizers often deal with a regularizers $g$, by computing the proximal map
\begin{equation}
  prox_g (\mathbf{x}) = \underset{\mathbf{u}}{argmin} \frac{1}{2}\vert\vert \mathbf{u}-\mathbf{x} \vert {\vert}^2 + g(\mathbf{x}).
\end{equation}

In order to implement those kinds of algorithms,RegularizedLeastSquares defines the following type
```julia
mutable struct Regularization <: AbstractRegularization
  prox!::Function
  norm::Function
  λ::Float64
  params::Dict{Symbol,Any}  # @TODO in die funcs
end
```
Here `prox!(x,λ)` is an in-place function which computes the proximal map on the input-vector `x`. The function `norm` computes the value of the corresponding term in the inverse problem and `λ` denotes the regularization parameter. If the Regularizer depends on additional parameters, those can be stored in `params`.

This design makes it possible to use the solvers in RegularizedLeastSquares.jl with custom regulizers without any overhead, besides implementing the proximal map.

## Implemented Regularizers
So far, the following common regularizers are implemented:
 * l$_1$ ("L1")
 * l$_2$ ("L2")
 * l$_{21}$ ("L21")
 * Locally Low Rank regularization ("LLR")
 * Nuclear Norm regularization ("Nuclear")
 * Positivity constrained ("Positive")
 * Projection onto a convex set ("Proj")

To build any of the implemted regularizers, one can use the methods `Regularization(name::String, λ::AbstractFloat; kargs...)` with the corresponding name (in brackets in the list above). For example, an $l_1$-regularizer can be build with
```julia
shape = (256,256) # size of the underlying Array
λ = 1.e-3         # regularization parameter
reg = Regularization("L1", λ; shape=shape)
```
