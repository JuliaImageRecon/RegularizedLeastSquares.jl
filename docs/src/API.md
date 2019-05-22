# API

## Solvers
```@docs
RegularizedLeastSquares.linearSolverList
RegularizedLeastSquares.createLinearSolver
```

## Regularization
```@docs
RegularizedLeastSquares.Regularization
RegularizedLeastSquares.Regularization(name::String, λ::AbstractFloat; kargs...)
RegularizedLeastSquares.Regularization(names::Vector{String}, λ::Vector{Float64}; kargs...)
RegularizedLeastSquares.RegularizationList
RegularizedLeastSquares.normalize!
RegularizedLeastSquares.proxL1!
RegularizedLeastSquares.proxL2!
RegularizedLeastSquares.proxL21!
RegularizedLeastSquares.proxLLR!
RegularizedLeastSquares.proxNuclear!
RegularizedLeastSquares.proxPositive!
RegularizedLeastSquares.proxProj!
RegularizedLeastSquares.proxTV!
RegularizedLeastSquares.normL1
RegularizedLeastSquares.normL2
RegularizedLeastSquares.normL21
RegularizedLeastSquares.normLLR
RegularizedLeastSquares.normNuclear
RegularizedLeastSquares.normPositive
RegularizedLeastSquares.normProj
RegularizedLeastSquares.normTV
```

## LinearOperators
```@docs
RegularizedLeastSquares.linearOperator(op::AbstractString, shape)
RegularizedLeastSquares.linearOperatorList
RegularizedLeastSquares.DCTOp
RegularizedLeastSquares.DSTOp
RegularizedLeastSquares.FFTOp
RegularizedLeastSquares.SamplingOp
RegularizedLeastSquares.WaveletOp
RegularizedLeastSquares.WeightingOp
```
