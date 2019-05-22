# Solvers

## Implemented Solvers
So far, RegularizedLeastSquares.jl provides implementations for the following solvers:
* Kaczrmarz algorithm ("kaczmarz")
* CGNR ("cgnr")
* Dax algorithm (with Kaczmarz) for unconstrained problems ("daxkaczmarz")
* Dax algorithm for constrained problems ("daxconstrained")
* Solver for the Fused Lasso problem ("fusedlasso")
* Fast Iterative Shrinkage Thresholding Algorithm ("fista")
* Alternating Direction of Multipliers Method ("admm")
* Split Bregman method for constrained inverse Problems ("splitBregman")

Here the strings given in brackets, denote the "name" of the respective solver in the repository.

## Creating a Solver
To create a solver, one can invoke the method `createLinearSolver` as in
```julia
solver = createLinearSolver("admm",A; reg=reg, ρ=0.1, iterations=20)
```
Here `A` denotes the system matrix and reg is either a `Regularization` or a`Vector{Regularization}`. All further solver parameters can be passed as keyword arguments. To make things more compact, it can be usefull to collect all parameters
in a `Dict{Symnbol,Any}`. In this way, the code snippet above can be written as
```julia
params=Dict{Symbol,Any}()
params[:reg] = reg
params[:ρ] = 0.1
params[:iterations] = 20

solver = createLinearSolver("admm",A; params...)
```
This notation can be convenient when a large number of parameters are set manually.
