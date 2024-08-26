# Solvers
RegularizedLeastSquares.jl provides a variety of solvers, which are used in fields such as MPI and MRI. The following is a non-exhaustive list of the implemented solvers:

* Kaczmarz algorithm (`Kaczmarz`, also called Algebraic reconstruction technique)
* Conjugate Gradients Normal Residual method (`CGNR`)
* Fast Iterative Shrinkage Thresholding Algorithm (`FISTA`)
* Alternating Direction of Multipliers Method (`ADMM`)

The solvers are organized in a type-hierarchy and inherit from:

```julia
abstract type AbstractLinearSolver
```

The type hierarchy is further differentiated into solver categories such as `AbstractRowAtionSolver`, `AbstractPrimalDualSolver` or `AbstractProximalGradientSolver`. A list of all available solvers can be returned by the `linearSolverList` function.

## Solver Construction
To create a solver, one can invoke the method `createLinearSolver` as in
```julia
solver = createLinearSolver(CGNR, A; reg=reg, kwargs...)
```
Here `A` denotes the operator and reg are the [Regularization](generated/explanations/regularization.md) terms to be used by the solver. All further solver parameters can be passed as keyword arguments and are solver specific. To make things more compact, it can be usefull to collect all parameters
in a `Dict{Symbol,Any}`. In this way, the code snippet above can be written as
```julia
params=Dict{Symbol,Any}()
params[:reg] = ...
...

solver = createLinearSolver(CGNR, A; params...)
```
This notation can be convenient when a large number of parameters are set manually.

It is also possible to construct a solver directly with its specific keyword arguments:

```julia
solver = CGNR(A, reg = reg, ...)
```

## Solver Usage
Once constructed, a solver can be used to approximate a solution to a given measurement vector:

```julia
x_approx = solve!(solver, b; kwargs...)
```
The keyword arguments can be used to supply an inital solution `x0`, one or more `callbacks` to interact and monitor the solvers state and more. See the How-To and the API for more information.

It is also possible to explicitly invoke the solvers iterations using Julias iterate interface:
```julia
init!(solver, b; kwargs...)
for (iteration, x_approx) in enumerate(solver)
    println("Iteration $iteration")
end
```
## Solver Internals
The fields of a solver can be divided into two groups. The first group are intended to be immutable fields that do not change during iterations, the second group are mutable fields that do change. Examples of the first group are the operator itself and examples of the second group are the current solution or the number of the current iteration.

The second group is usually encapsulated in its own state struct:
```julia
mutable struct Solver{matT, ...}
  A::matT
  # Other "static" fields
  state::AbstractSolverState{<:Solver}
end

mutable struct SolverState{T, tempT} <: AbstractSolverState{Solver}
  x::tempT
  rho::T
  # ...
  iteration::Int64
end
```
States are subtypes of the parametric `AbstractSolverState{S}` type. The state fields of solvers can be exchanged with different state belonging to the correct solver `S`. This means that the states can be used to realize custom variants of an existing solver:
```julia
mutable struct VariantState{T, tempT} <: AbstractSolverState{Solver}
  x::tempT
  other::tempT
  # ...
  iteration::Int64
end

SolverVariant(A; kwargs...) = Solver(A, VariantState(kwargs...))

function iterate(solver::Solver, state::VarianteState)
  # Custom iteration
end
```