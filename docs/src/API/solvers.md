# API for Solvers
This page contains documentation of the public API of the RegularizedLeastSquares. In the Julia
REPL one can access this documentation by entering the help mode with `?`

## solve!
```@docs
RegularizedLeastSquares.solve!(::AbstractLinearSolver, ::Any)
```

## ADMM
```@docs
RegularizedLeastSquares.ADMM
```

## CGNR
```@docs
RegularizedLeastSquares.CGNR
```

## Kaczmarz
```@docs
RegularizedLeastSquares.Kaczmarz
```

## FISTA
```@docs
RegularizedLeastSquares.FISTA
```

## OptISTA
```@docs
RegularizedLeastSquares.OptISTA
```

## POGM
```@docs
RegularizedLeastSquares.POGM
```

## SplitBregman
```@docs
RegularizedLeastSquares.SplitBregman
```

## Miscellaneous Functions
```@docs
RegularizedLeastSquares.linearSolverList
RegularizedLeastSquares.createLinearSolver
RegularizedLeastSquares.applicableSolverList
RegularizedLeastSquares.isapplicable
```