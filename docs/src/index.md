# RegularizedLeastSquares.jl

*Solvers for Linear Inverse Problems using Regularization Techniques*

## Introduction

RegularizedLeastSquares.jl is a Julia package for solving large scale linear systems
using different types of algorithms. Ill-conditioned problems arise in many areas of practical interest. To solve these problems, one often resorts to regularization techniques and non-linear problem formulations. This packages provides implementations for a variety of solvers, which are used in fields such as MPI and MRI.

The implemented methods range from the $l^2_2$-regularized CGNR method to more general optimizers such as the Alternating Direction of Multipliers Method (ADMM) or the Split-Bregman method.

For convenience, implementations of popular regularizers, such as $l_1$-regularization and TV regularization, are provided. On the other hand, hand-crafted regularizers can be used quite easily.

Depending on the problem, it becomes unfeasible to store the full system matrix at hand. For this purpose, RegularizedLeastSquares.jl allows for the use of matrix-free operators. Such operators can be realized using the interface provided by the package [LinearOperators.jl](https://github.com/JuliaSmoothOptimizers/LinearOperators.jl).
Other interfaces can be used as well, as long as the product `*(A,x)` and the adjoint `adjoint(A)` are provided. A number of common matrix-free operators are provided by the package [LinearOperatorColection.jl](https://github.com/JuliaImageRecon/LinearOperatorCollection.jl).

## Usage

  * See [Getting Started](@ref) for an introduction to using the package
