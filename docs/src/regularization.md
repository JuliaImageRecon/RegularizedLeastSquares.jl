```@meta
DocTestSetup = quote
    using RegularizedLeastSquares, Wavelets, LinearOperatorCollection
end
```
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

```jldoctest l2
julia> l2 = L2Regularization(0.3)
L2Regularization{Float64}(0.3)
```
Parameterized regularization terms implement:
```julia
prox!(reg::AbstractParameterizedRegularization, x, λ)
norm(reg::AbstractParameterizedRegularization, x, λ)
```
where `λ` by default is filled with the value used during construction.

Invoking `λ` on a parameterized term retrieves its regularization parameter. This can be used in a solver to scale and overwrite the parameter as follows:
```jldoctest l2
julia> prox!(l2, [1.0])
1-element Vector{Float64}:
 0.625

julia> param = λ(l2)
0.3

julia> prox!(l2, [1.0], param*0.2)
1-element Vector{Float64}:
 0.8928571428571428

```

## Projection Regularization Terms
This group of regularization terms implement projections, such as a positivity constraint or a projection with a given convex projection function.

```jldoctest pos
julia> positive = PositiveRegularization()
PositiveRegularization()

julia> prox!(positive, [2.0, -0.2])
2-element Vector{Float64}:
 2.0
 0.0
```

## Nested Regularization Terms
Nested regularization terms are terms that act as decorators to the core regularization terms. These terms can be nested around other terms and add functionality to a regularization term, such as scaling `λ` based on the provided system matrix or applying a transform, such as the Wavelet, to `x`:

```jldoctest wavelet
julia> core = L1Regularization(0.8)
L1Regularization{Float64}(0.8)

julia> wop = WaveletOp(Float32, shape = (32,32));

julia> reg = TransformedRegularization(core, wop);

julia> prox!(reg, randn(32*32)); # Apply soft-thresholding in Wavelet domain
```
The type of regularization term a nested term can be wrapped around depends on the concrete type of the nested term. However generally, they can be nested arbitrarly deep, adding new functionality with each layer. Each nested regularization term can return its `inner` regularization. Furthermore, all regularization terms implement the iteration interface to iterate over the nesting. The innermost regularization term of a nested term must be a core regularization term and it can be returned by the `sink` function:
```jldoctest wavelet
julia> innerreg(reg) == core
true

julia> sink(reg) == core
true

julia> foreach(r -> println(nameof(typeof(r))), reg)
TransformedRegularization
L1Regularization
```