module RegularizedLeastSquares

import Base: length, iterate
using LinearAlgebra
import LinearAlgebra.BLAS: gemv, gemv!
import LinearAlgebra: BlasFloat, normalize!, norm, rmul!, lmul!
using SparseArrays
using IterativeSolvers
using Random
using VectorizationBase
using VectorizationBase: shufflevector, zstridedpointer
using FLoops
#@reexport using SparsityOperators
using SparsityOperators: normalOperator, opEye
using ProgressMeter
using StatsBase

export createLinearSolver, init, deinit, solve, linearSolverList,linearSolverListReal

abstract type AbstractLinearSolver end
# The following is just for documentation purposes. To allow for different operator
# libraries we allow the Trafo to be of type Any.
const Trafo = Any # Union{AbstractMatrix, AbstractLinearOperator, Nothing}

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
include("proximalMaps/ProxTVCondat.jl")
include("proximalMaps/ProxNuclear.jl")

include("Regularization.jl")
include("Utils.jl")
include("Kaczmarz.jl")
include("KaczmarzUpdated.jl")
include("DAXKaczmarz.jl")
include("DAXConstrained.jl")
include("CGNR.jl")
include("Direct.jl")
include("FusedLasso.jl")
include("FISTA.jl")
include("ADMM.jl")
include("SplitBregman.jl")
include("PrimalDualSolver.jl")

"""
Return a list of all available linear solvers
"""
function linearSolverList()
  Any["kaczmarz","cgnr"] # These are those passing the tests
    #, "fusedlasso"]
end

function linearSolverListReal()
  Any["kaczmarzUpdated","kaczmarz","cgnr","daxkaczmarz","daxconstrained","primaldualsolver"] # These are those passing the tests
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
* `"primaldualsolver"`- First order primal dual method
"""
function createLinearSolver(solver::AbstractString, A, x=zeros(eltype(A),size(A,2));
                            log::Bool=false, kargs...)

  log ? solverInfo = SolverInfo(;kargs...) : solverInfo=nothing

  if solver == "kaczmarz"
    return Kaczmarz(A; kargs...)
  elseif solver == "kaczmarzUpdated"
    return KaczmarzUpdated(A; kargs...)
  elseif solver == "cgnr"
    return CGNR(A, x; kargs...)
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
    return FISTA(A, x; kargs...)
  elseif solver == "admm"
    return ADMM(A, x; kargs...)
  elseif solver == "splitBregman"
    return SplitBregman(A, x; kargs...)
  elseif solver == "primaldualsolver"
    return PrimalDualSolver(A; kargs...)
  else
    error("Solver $solver not found.")
    return Kaczmarz(A; kargs...)
  end
end

end
