module LinearSolver

using ProgressMeter, IterativeSolvers

import Base.LinAlg: A_mul_B!, Ac_mul_B!, At_mul_B!, length
using Base.LinAlg: BlasFloat

export createLinearSolver, init, deinit, solve, linearSolverList,linearSolverListReal

abstract AbstractLinearSolver

# Fallback function
setlambda(S::AbstractMatrix, λ::Real) = nothing

include("LazyMatrixTranspose.jl")
include("LinearOperator.jl")
include("Utils.jl")
include("Kaczmarz.jl")
include("DAX.jl")
include("CGNR.jl")
include("Direct.jl")
include("FusedLasso.jl")
include("FISTA.jl")

"""
Return a list of all available linear solvers
"""
function linearSolverList()
  Any["kaczmarz","fista"] #"cgnr"  # These are those passing the tests
    #, "fusedlasso"]
end

function linearSolverListReal()
  Any["kaczmarz","daxkaczmarz","daxconstrained"] # These are those passing the tests
    #, "fusedlasso"] "cgnr"
end


@doc """
This file contains linear solver that are commonly used in MPI
Currently implemented are
 - kaczmarz method (the default)
 - CGNR
 - A direct solver using the \ operator


All solvers return an approximate solution to Sᵀx = u.


Function returns choosen solver.
""" ->
function createLinearSolver(solver::AbstractString, A; kargs...)
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
  elseif solver == "lsqr"
    return LSQR(A; kargs...)
  elseif solver == "pseudoinverse"
    return PseudoInverse(A; kargs...)
  elseif solver == "fusedlasso"
    return FusedLasso(A; kargs...)
  elseif solver == "fista"
    return FISTA(A; kargs...)
  else
    error("Solver $solver not found.")
  end
end


end
