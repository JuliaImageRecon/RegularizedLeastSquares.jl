module RegularizedLeastSquares

import Base: length
using FFTW
using LinearAlgebra
import LinearAlgebra.BLAS: gemv, gemv!
import LinearAlgebra: BlasFloat, normalize!, norm, rmul!, lmul!
using LinearOperators
using SparseArrays
using ProgressMeter
using IterativeSolvers
using Random

export createLinearSolver, init, deinit, solve, linearSolverList,linearSolverListReal

abstract type AbstractLinearSolver end
const Trafo = Union{AbstractMatrix, AbstractLinearOperator, Nothing}
const FuncOrNothing = Union{Function, Nothing}

# Fallback function
setlambda(S::AbstractMatrix, λ::Real) = nothing

include("proximalMaps/ProxL1.jl")
include("proximalMaps/ProxL2.jl")
include("proximalMaps/ProxL21.jl")
include("proximalMaps/ProxLLR.jl")
# include("proximalMaps/ProxSLR.jl")
include("proximalMaps/ProxPositive.jl")
include("proximalMaps/ProxProj.jl")
include("proximalMaps/ProxTV.jl")
include("proximalMaps/ProxNuclear.jl")

include("Regularization.jl")
include("LinearOperator.jl")
include("Utils.jl")
include("Kaczmarz.jl")
include("DAX.jl")
include("CGNR.jl")
include("CG.jl")
include("Direct.jl")
include("FusedLasso.jl")
include("FISTA.jl")
include("ADMM.jl")
include("SplitBregman.jl")

"""
Return a list of all available linear solvers
"""
function linearSolverList()
  Any["kaczmarz","cgnr"] # These are those passing the tests
    #, "fusedlasso"]
end

function linearSolverListReal()
  Any["kaczmarz","cgnr","daxkaczmarz","daxconstrained"] # These are those passing the tests
    #, "fusedlasso"]
end


"""
    createLinearSolver(solver::AbstractString, A; log::Bool=false, kargs...)

This method creates a solver. The supported solvers are methods typically used in MPI/MRI.
All solvers return an approximate solution to STx = u.
Function returns choosen solver.

# solvers:
* `"kaczmarz"`        - kaczmarz method (the default)
* `"cgnr`             - CGNR
* `"direct"`          - A direct solver using the backslash operator
* `"daxkaczmarz"`     - Dax algorithm (with Kaczmarz) for unconstrained problems
* `"daxconstrained"`  - Dax algorithm for constrained problems
* `"pseudoinverse"`   - approximates a solution using the More-Penrose pseudo inverse
* `"fusedlasso"`      - solver for the Fused-Lasso problem
* `"fista"`           - Fast Iterative Shrinkage Thresholding Algorithm
* `"admm"`            - Alternating Direcion of Multipliers Method
* `"splitBregman"`    - Split Bregman method for constrained & regularized inverse problems
"""
function createLinearSolver(solver::AbstractString, A;
                            log::Bool=false, kargs...)

  log ? solverInfo = SolverInfo(;kargs...) : solverInfo=nothing

  if solver == "kaczmarz"
    return Kaczmarz(A; kargs...)
  elseif solver == "cgnr"
    return CGNR(A; kargs...)
  elseif solver == "direct"
    return DirectSolver(A; kargs...)
  elseif solver == "daxkaczmarz"
    return DaxKaczmarz(A; kargs...)
  elseif solver == "daxconstrained"
    return DaxConstrained(A; kargs...)
  elseif solver == "pseudoinverse"
    return PseudoInverse(A; kargs...)
  elseif solver == "fusedlasso"
    return FusedLasso(A; kargs...)
  elseif solver == "fista"
    return FISTA(A; kargs...)
  elseif solver == "admm"
    return ADMM(A; kargs...)
  elseif solver == "splitBregman"
    return SplitBregman(A; kargs...)
  else
    error("Solver $solver not found.")
    return Kaczmarz(A; kargs...)
  end
end

end
