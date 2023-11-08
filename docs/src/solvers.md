# Solvers
RegularizedLeastSquares.jl provides a variety of solvers, which are used in fields such as MPI and MRI. The following is a non-exhaustive list of the implemented solvers:

* Kaczmarz algorithm (`Kaczmarz`)
* Conjugate Gradients Normal Residual method (`CGNR`)
* Fast Iterative Shrinkage Thresholding Algorithm (`FISTA`)
* Alternating Direction of Multipliers Method (`ADMM`)

The solvers are organized in a type-hierarchy and inherit from:

```julia
abstract type AbstractLinearSolver
```

The type hierarchy is further differentiated into solver categories such as `AbstractRowAtionSolver`, `AbstractPrimalDualSolver` or `AbstractProximalGradientSolver`. A list of all available solvers can be returned by the `linearSolverList` function.
## Creating a Solver
To create a solver, one can invoke the method `createLinearSolver` as in
```julia
solver = createLinearSolver(ADMM, A; reg=reg, kwargs...)
```
Here `A` denotes the system matrix and reg are the [Regularization](regularization.md) terms to be used by the solver. All further solver parameters can be passed as keyword arguments and are solver specific. To make things more compact, it can be usefull to collect all parameters
in a `Dict{Symbol,Any}`. In this way, the code snippet above can be written as
```julia
params=Dict{Symbol,Any}()
params[:reg] = ...
...

solver = createLinearSolver(ADMM, A; params...)
```
This notation can be convenient when a large number of parameters are set manually.

It is possible to check if a given solver is applicable to the wanted arguments, as not all solvers are applicable to all system matrix and data (element) types or regularization terms combinations. This is achieved with the `isapplicable` function:

```julia
isapplicable(Kaczmarz, A, x, [L21Regularization(0.4f0)])
false
```

For a given set of arguments the list of applicable solvers can be retrieved with `applicableSolverList`.