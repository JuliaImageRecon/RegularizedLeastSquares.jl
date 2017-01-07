module LinearSolver

using ProgressMeter, IterativeSolvers

import Base.LinAlg: A_mul_B!, Ac_mul_B!, At_mul_B! #import to add new method
using Base.LinAlg: BlasFloat

export linearSolver

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
include("LSQR.jl")
include("FusedLasso.jl")


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
  else
    error("Solver $solver not found.")
  end
end

function linearSolver(solver::AbstractString)
  depwarn("function linearSolver is deprectated use createLinearSolver instead", :linearSolver)
  if solver == "cgnr"
    return cgnr
  elseif solver == "direct"
    return directSolver
  elseif solver == "kaczmarzold"
    return kaczmarzold
  elseif solver == "tobitestcgnr"
    return tobitestcgnr
  elseif solver == "daxkaczmarz"
    return daxrandkaczmarz
  elseif solver == "daxconstrained"
    return daxconstrained
  elseif solver == "kaczmarz"
    return kaczmarz
  elseif solver == "lsqr"
    return lsqr_
  elseif solver == "pseudoinverse"
    return pseudoinverse
  else
    error("Solver ",solver," not found.")
  end
end

end
