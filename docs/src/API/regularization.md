# API for Regularizers
This page contains documentation of the public API of the RegularizedLeastSquares. In the Julia
REPL one can access this documentation by entering the help mode with `?`

```@docs
RegularizedLeastSquares.L1Regularization
RegularizedLeastSquares.SqrL2Regularization
RegularizedLeastSquares.L21Regularization
RegularizedLeastSquares.LLRRegularization
RegularizedLeastSquares.NuclearRegularization
RegularizedLeastSquares.TVRegularization
```

## Projection Regularization
```@docs
RegularizedLeastSquares.PositiveRegularization
RegularizedLeastSquares.RealRegularization
```

## Nested Regularization
```@docs
RegularizedLeastSquares.innerreg(::AbstractNestedRegularization)
RegularizedLeastSquares.sink(::AbstractNestedRegularization)
RegularizedLeastSquares.sinktype(::AbstractNestedRegularization)
```

## Scaled Regularization
```@docs
RegularizedLeastSquares.AbstractScaledRegularization
RegularizedLeastSquares.scalefactor
RegularizedLeastSquares.NormalizedRegularization
RegularizedLeastSquares.NoNormalization
RegularizedLeastSquares.MeasurementBasedNormalization
RegularizedLeastSquares.SystemMatrixBasedNormalization
RegularizedLeastSquares.FixedParameterRegularization
```

## Misc. Nested Regularization
```@docs
RegularizedLeastSquares.MaskedRegularization
RegularizedLeastSquares.TransformedRegularization
RegularizedLeastSquares.PlugAndPlayRegularization
```

## Miscellaneous Functions
```@docs
RegularizedLeastSquares.prox!(::AbstractParameterizedRegularization, ::AbstractArray)
RegularizedLeastSquares.prox!(::Type{<:AbstractParameterizedRegularization}, ::Any, ::Any)
RegularizedLeastSquares.norm(::AbstractParameterizedRegularization, ::AbstractArray)
RegularizedLeastSquares.λ(::AbstractParameterizedRegularization)
RegularizedLeastSquares.norm(::Type{<:AbstractParameterizedRegularization}, ::Any, ::Any)
```