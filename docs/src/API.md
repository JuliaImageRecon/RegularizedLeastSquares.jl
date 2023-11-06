# API
This page contains documentation of the public API of the RegularizedLeastSquares. In the Julia
REPL one can access this documentation by entering the help mode with `?`

## Solvers
```@docs
RegularizedLeastSquares.linearSolverList
RegularizedLeastSquares.createLinearSolver
RegularizedLeastSquares.solve
RegularizedLeastSquares.SolverInfo
RegularizedLeastSquares.ADMM
RegularizedLeastSquares.CGNR
RegularizedLeastSquares.Kaczmarz
RegularizedLeastSquares.FISTA
RegularizedLeastSquares.OptISTA
RegularizedLeastSquares.POGM
RegularizedLeastSquares.SplitBregman
```

## Regularization
```@docs
RegularizedLeastSquares.prox!(::AbstractParameterizedRegularization, ::AbstractArray)
RegularizedLeastSquares.norm(::AbstractParameterizedRegularization, ::AbstractArray)
RegularizedLeastSquares.λ(::AbstractParameterizedRegularization)
RegularizedLeastSquares.prox!(::Type{<:AbstractParameterizedRegularization}, ::Any, ::Any)
RegularizedLeastSquares.norm(::Type{<:AbstractParameterizedRegularization}, ::Any, ::Any)
```
### Parameterized Regularization
```@docs
RegularizedLeastSquares.L1Regularization
RegularizedLeastSquares.L2Regularization
RegularizedLeastSquares.L21Regularization
RegularizedLeastSquares.LLRRegularization
RegularizedLeastSquares.NuclearRegularization
RegularizedLeastSquares.TVRegularization
```
### Projection Regularization
```@docs
RegularizedLeastSquares.PositiveRegularization
RegularizedLeastSquares.RealRegularization
```
## Nested Regularization
```@docs
RegularizedLeastSquares.nested(::AbstractNestedRegularization)
RegularizedLeastSquares.sink(::AbstractNestedRegularization)
RegularizedLeastSquares.sinktype(::AbstractNestedRegularization)
```
### Scaled Regularization
```@docs
RegularizedLeastSquares.AbstractScaledRegularization
RegularizedLeastSquares.factor
RegularizedLeastSquares.NormalizedRegularization
RegularizedLeastSquares.NoNormalization
RegularizedLeastSquares.MeasurementBasedNormalization
RegularizedLeastSquares.SystemMatrixBasedNormalization
```
### Misc. Nested Regularization
```@docs
RegularizedLeastSquares.MaskedRegularization
RegularizedLeastSquares.TransformedRegularization
RegularizedLeastSquares.ConstraintTransformedRegularization
```