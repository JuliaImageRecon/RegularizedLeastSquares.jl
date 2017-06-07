module LinearSolver

using ProgressMeter, IterativeSolvers

import Base.LinAlg: A_mul_B!, Ac_mul_B!, At_mul_B!, length
using Base.LinAlg: BlasFloat

export createLinearSolver, init, deinit, solve, linearSolverList,linearSolverListReal

abstract AbstractLinearSolver

# Fallback function
setlambda(S::AbstractMatrix, λ::Real) = nothing

include("Regularization.jl")
include("LazyMatrixTranspose.jl")
include("LinearOperator.jl")
include("Utils.jl")
include("Kaczmarz.jl")
include("DAX.jl")
include("CGNR.jl")
include("Direct.jl")
include("LSQR.jl")
include("FusedLasso.jl")

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


@doc """
This file contains linear solver that are commonly used in MPI
Currently implemented are
 - kaczmarz method (the default)
 - CGNR
 - A direct solver using the \ operator


All solvers return an approximate solution to Sᵀx = u.


Function returns choosen solver.
""" ->
function createLinearSolver(solver::AbstractString, regParams, solverParams)

  reg = Regularization(;regParams...)

  if solver == "kaczmarz"
    return Kaczmarz(A; regularizer=reg, solverParams...)
  elseif solver == "cgnr"
    return CGNR(A; regularizer=reg, solverParams...)
  elseif solver == "direct"
    return DirectSolver(A; solverParams...)
  elseif solver == "daxkaczmarz"
    return DaxKaczmarz(A; solverParams...)
  elseif solver == "daxconstrained"
    return DaxConstrained(A; solverParams...)
  elseif solver == "lsqr"
    return LSQR(A; solverParams...)
  elseif solver == "pseudoinverse"
    return PseudoInverse(A; solverParams...)
  elseif solver == "fusedlasso"
    return FusedLasso(A; regularizer=reg, solverParams...)
  else
    error("Solver $solver not found.")
  end
end


end
