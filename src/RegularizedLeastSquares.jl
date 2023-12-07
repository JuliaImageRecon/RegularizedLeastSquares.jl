module RegularizedLeastSquares

import Base: length, iterate, findfirst, copy
using LinearAlgebra
import LinearAlgebra.BLAS: gemv, gemv!
import LinearAlgebra: BlasFloat, normalize!, norm, rmul!, lmul!
using SparseArrays
using IterativeSolvers
using Random
using VectorizationBase
using VectorizationBase: shufflevector, zstridedpointer
using FLoops
using LinearOperators: opEye
using StatsBase
using LinearOperatorCollection
using InteractiveUtils

export AbstractLinearSolver, createLinearSolver, init, deinit, solve!, linearSolverList, linearSolverListReal, applicableSolverList

abstract type AbstractLinearSolver end

"""
    solve!(solver::AbstractLinearSolver, b; x0 = 0, callback = (_, _) -> nothing)

Solves an inverse problem for the data vector `b` using `solver`.

# Required Arguments
  * `solver::AbstractLinearSolver`    - linear solver (e.g., `ADMM` or `FISTA`), containing forward/normal operator and regularizer
  * `b::AbstractVector`               - data vector if `A` was supplied to the solver, back-projection of the data otherwise

# Optional Keyword Arguments
  * `x0::AbstractVector`              - initial guess for the solution; default is zero
  * `callback::Function`               - function that takes the two arguments `callback(solver, iteration)` and, e.g., stores, prints, or plots the intermediate solutions or convergence parameters. Be sure not to modify `solver` or `iteration` in the callback function as this would japaridze convergence.


# Examples
The optimization problem
```math
	argmin_x ||Ax - b||_2^2 + λ ||x||_1
```
can be solved with the following lines of code:
```jldoctest solveExample
julia> using RegularizedLeastSquares

julia> A = [0.831658  0.96717
            0.383056  0.39043
            0.820692  0.08118];

julia> x = [0.5932234523399985; 0.2697534345340015];

julia> b = A * x;

julia> S = ADMM(A);

julia> x_approx = solve!(S, b)
2-element Vector{Float64}:
 0.5932234523399984
 0.26975343453400163
```
Here, we use [`L1Regularization`](@ref), which is default for [`ADMM`](@ref). All regularization options can be found in [API for Regularizers](@ref).

The following example solves the same problem, but stores the solution `x` of each interation in `tr`:
```jldoctest solveExample
julia> tr = Dict[]
Dict[]

julia> store_trace!(tr, solver, iteration) = push!(tr, Dict("iteration" => iteration, "x" => solver.x, "beta" => solver.β))
store_trace! (generic function with 1 method)

julia> x_approx = solve!(S, b; callback=(solver, iteration) -> store_trace!(tr, solver, iteration))
2-element Vector{Float64}:
 0.5932234523399984
 0.26975343453400163

julia> tr[3]
Dict{String, Any} with 3 entries:
  "iteration" => 2
  "x"         => [0.593223, 0.269753]
  "beta"      => [1.23152, 0.927611]
```

The last example show demonstrates how to plot the solution at every 10th iteration:
```julia
julia> using Plots

julia> function plot_trace(solver, iteration)
         if iteration % 10 == 0
           display(scatter(solver.x))
         end
       end
plot_trace (generic function with 1 method)

julia> x_approx = solve!(S, b; callback = plot_trace);
```
The keyword `callback` allows you to pass any function that takes the arguments `solver` and `iteration` and prints, stores, or plots intermediate result.
"""
function solve!(solver::AbstractLinearSolver, b; x0 = 0, callback = (_, _) -> nothing)
  init!(solver, b; x0)
  callback(solver, 0)

  for (iteration, _) = enumerate(solver)
    callback(solver, iteration)
  end

  return solver.x
end



export AbstractRowActionSolver
abstract type AbstractRowActionSolver <: AbstractLinearSolver end

export AbstractDirectSolver
abstract type AbstractDirectSolver <: AbstractLinearSolver end

export AbstractPrimalDualSolver
abstract type AbstractPrimalDualSolver <: AbstractLinearSolver end

export AbstractProximalGradientSolver
abstract type AbstractProximalGradientSolver <: AbstractLinearSolver end

export AbstractKrylovSolver
abstract type AbstractKrylovSolver <: AbstractLinearSolver end

# Fallback function
setlambda(S::AbstractMatrix, λ::Real) = nothing

include("Transforms.jl")
include("Regularization/Regularization.jl")
include("proximalMaps/ProximalMaps.jl")

include("Utils.jl")
include("Kaczmarz.jl")
include("DAXKaczmarz.jl")
include("DAXConstrained.jl")
include("CGNR.jl")
include("Direct.jl")
include("FISTA.jl")
include("OptISTA.jl")
include("POGM.jl")
include("ADMM.jl")
include("SplitBregman.jl")
include("PrimalDualSolver.jl")

"""
Return a list of all available linear solvers
"""
function linearSolverList()
  filter(s -> s ∉ [DaxKaczmarz, DaxConstrained, PrimalDualSolver], linearSolverListReal())
end

function linearSolverListReal()
  union(subtypes.(subtypes(AbstractLinearSolver))...) # For deeper nested type extend this to loop for types with isabstracttype == true
end

export isapplicable
isapplicable(solver::AbstractLinearSolver, args...) = isapplicable(typeof(solver), args...)
isapplicable(x, reg::AbstractRegularization) = isapplicable(x, [reg])
isapplicable(::Type{T}, reg::Vector{<:AbstractRegularization}) where T <: AbstractLinearSolver = false

function isapplicable(::Type{T}, reg::Vector{<:AbstractRegularization}) where T <: AbstractRowActionSolver
  applicable = true
  applicable &= length(findsinks(AbstractParameterizedRegularization, reg)) <= 2
  applicable &= length(findsinks(L2Regularization, reg)) == 1
  return applicable
end

function isapplicable(::Type{T}, reg::Vector{<:AbstractRegularization}) where T <: AbstractPrimalDualSolver
  # TODO
  return true
end

function isapplicable(::Type{T}, reg::Vector{<:AbstractRegularization}) where T <: AbstractProximalGradientSolver
  applicable = true
  applicable &= length(findsinks(AbstractParameterizedRegularization, reg)) == 1
  return applicable
end

function isapplicable(::Type{T}, A, x) where T <: AbstractLinearSolver
  # TODO
  applicable = true
  return applicable
end

"""
    isapplicable(solverType::Type{<:AbstractLinearSolver}, A, x, reg)

return `true` if a `solver` of type `solverType` is applicable to system matrix `A`, data `x` and regularization terms `reg`.
"""
isapplicable(::Type{T}, A, x, reg) where T <: AbstractLinearSolver = isapplicable(T, A, x) && isapplicable(T, reg)

"""
    applicable(args...)

list all `solvers` that are applicable to the given arguments. Arguments are the same as for `isapplicable` without the `solver` type.

See also [`isapplicable`](@ref), [`linearSolverList`](@ref).
"""
applicableSolverList(args...) = filter(solver -> isapplicable(solver, args...), linearSolverListReal())

"""
    createLinearSolver(solver::AbstractLinearSolver, A; kargs...)

This method creates a solver. The supported solvers are methods typically used for solving
regularized linear systems. All solvers return an approximate solution to Ax = b.

TODO: give a hint what solvers are available
"""
function createLinearSolver(solver::Type{T}, A; kargs...) where {T<:AbstractLinearSolver}
  return solver(A; kargs...)
end

@deprecate createLinearSolver(solver, A, x; kargs...) createLinearSolver(solver, A; kargs...)

end
