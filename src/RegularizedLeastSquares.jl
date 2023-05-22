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
using LinearOperators: opEye
using ProgressMeter
using StatsBase
using FastClosures
using LinearOperators

export AbstractLinearSolver, createLinearSolver, init, deinit, solve, linearSolverList,linearSolverListReal

abstract type AbstractLinearSolver end
# The following is just for documentation purposes. To allow for different operator
# libraries we allow the Trafo to be of type Any.
const Trafo = Any # Union{AbstractMatrix, AbstractLinearOperator, Nothing}

# Fallback function
setlambda(S::AbstractMatrix, λ::Real) = nothing

include("linearOperators/GradientOp.jl")

include("Regularization.jl")
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
function createLinearSolver(solver::AbstractString, A, x=zeros(eltype(A),size(A,2)); kargs...)

  if solver == "kaczmarz"
    return createLinearSolver(Kaczmarz, A; kargs...)
  elseif solver == "kaczmarzUpdated"
    return createLinearSolver(KaczmarzUpdated, A; kargs...)
  elseif solver == "cgnr"
    return createLinearSolver(CGNR, A, x; kargs...)
  elseif solver == "direct"
    return createLinearSolver(DirectSolver, A; kargs...)
  elseif solver == "daxkaczmarz"
    return createLinearSolver(DaxKaczmarz, A; kargs...)
  elseif solver == "daxconstrained"
    return createLinearSolver(DaxConstrained, A; kargs...)
  elseif solver == "pseudoinverse"
    return createLinearSolver(PseudoInverse, A; kargs...)
  elseif solver == "fusedlasso"
    return createLinearSolver(FusedLasso, A; kargs...)
  elseif solver == "fista"
    return createLinearSolver(FISTA, A, x; kargs...)
  elseif solver == "admm"
    return createLinearSolver(ADMM, A, x; kargs...)
  elseif solver == "splitBregman"
    return createLinearSolver(SplitBregman, A, x; kargs...)
  elseif solver == "primaldualsolver"
    return createLinearSolver(PrimalDualSolver, A; kargs...)
  else
    error("Solver $solver not found.")
    return Kaczmarz(A; kargs...)
  end
end

function createLinearSolver(solver::Type{T}, A; log::Bool=false, kargs...) where {T<:AbstractLinearSolver}
  log ? solverInfo = SolverInfo(;kargs...) : solverInfo=nothing
  return solver(A; kargs...)
end

function createLinearSolver(solver::Type{T}, A, x; log::Bool=false, kargs...) where {T<:AbstractLinearSolver}
  log ? solverInfo = SolverInfo(;kargs...) : solverInfo=nothing
  return solver(A, x; kargs...)
end


end
