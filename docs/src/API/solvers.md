# API
This page contains documentation of the public API of the RegularizedLeastSquares. In the Julia
REPL one can access this documentation by entering the help mode with `?`

## General
```@docs
RegularizedLeastSquares.linearSolverList
RegularizedLeastSquares.createLinearSolver
RegularizedLeastSquares.solve(::AbstractLinearSolver, ::Any)
RegularizedLeastSquares.SolverInfo
RegularizedLeastSquares.applicableSolverList
RegularizedLeastSquares.isapplicable
```

## ADMM
```@docs
RegularizedLeastSquares.ADMM
RegularizedLeastSquares.solve(::ADMM, ::Any)
```

## CGNR
```@docs
RegularizedLeastSquares.CGNR
RegularizedLeastSquares.solve(::CGNR, ::Any)
```

## Kaczmarz
```@docs
RegularizedLeastSquares.Kaczmarz
RegularizedLeastSquares.solve(::Kaczmarz, ::Vector{Any})
```

## FISTA
```@docs
RegularizedLeastSquares.FISTA
RegularizedLeastSquares.solve(::FISTA, ::Any)
```

## OptISTA
```@docs
RegularizedLeastSquares.OptISTA
RegularizedLeastSquares.solve(::OptISTA, ::Any)
```

## POGM
```@docs
RegularizedLeastSquares.POGM
RegularizedLeastSquares.solve(::POGM, ::Any)
```

## SplitBregman
```@docs
RegularizedLeastSquares.SplitBregman
RegularizedLeastSquares.solve(::SplitBregman, ::Any)
```

## Direct