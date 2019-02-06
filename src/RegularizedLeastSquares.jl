module RegularizedLeastSquares

import Base: length
using FFTW
using LinearAlgebra
import LinearAlgebra.BLAS: gemv, gemv!
import LinearAlgebra: BlasFloat, normalize!, norm, rmul!, lmul!
using LinearOperators
using SparseArrays
using ProgressMeter

export createLinearSolver, init, deinit, solve, linearSolverList,linearSolverListReal

abstract type AbstractLinearSolver end
const Trafo = Union{AbstractMatrix, AbstractLinearOperator, Nothing}

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
This file contains linear solver that are commonly used in MPI
Currently implemented are
 - kaczmarz method (the default)
 - CGNR
 - A direct solver using the backslash operator


All solvers return an approximate solution to STx = u.


Function returns choosen solver.
"""
function createLinearSolver(solver::AbstractString, A, reg=nothing; lambdL2::Float64=0.0, λ::Float64=0.0, kargs...)

  # hack for compatibility with MPIReco.jl -> FIXME
  if lambdL2 != 0
     λ = lambdL2
  end

  if solver == "kaczmarz"
    reg =  getRegularization("L2",λ)
    return Kaczmarz(A, reg; kargs...)
  elseif solver == "cgnr"
    reg =  getRegularization("L2",λ)
    return CGNR(A, reg; kargs...)
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
    reg==nothing ? reg = Regularization(;kargs...) : nothing
    return FISTA(A, reg;kargs...)
  elseif solver == "admm"
    reg==nothing ? reg = Regularization(;kargs...) : nothing
    return ADMM(A, reg;kargs...)
  else
    error("Solver $solver not found.")
  end
end

end
