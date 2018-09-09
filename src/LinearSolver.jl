module LinearSolver

using ProgressMeter, Compat

using LinearAlgebra
using LinearOperators

if VERSION < v"1.0.0"
  import LinearAlgebra: Ac_mul_B!, At_mul_B!
end
import LinearOperators: A_mul_B!

import LinearAlgebra: BlasFloat, normalize!, norm
import Base: length

using SparseArrays
using Random
using FFTW


export createLinearSolver, init, deinit, solve, linearSolverList,linearSolverListReal

@compat abstract type AbstractLinearSolver end

# Fallback function
setlambda(S::AbstractMatrix, λ::Real) = nothing

include("Regularization.jl")

include("proximalMaps/ProxL1.jl")
include("proximalMaps/ProxL2.jl")
include("proximalMaps/ProxL21.jl")
include("proximalMaps/ProxLLR.jl")
include("proximalMaps/ProxSLR.jl")
include("proximalMaps/ProxPositive.jl")
include("proximalMaps/ProxProj.jl")
include("proximalMaps/ProxTV.jl")
include("proximalMaps/ProxNuclear.jl")

include("LazyMatrixTranspose.jl")
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
function createLinearSolver(solver::AbstractString, A, reg=nothing; kargs...)

  reg==nothing ? reg = Regularization(;kargs...) : nothing

  if solver == "kaczmarz"
    for regEntry in RegularizationList()
      regEntry == "L2" ? continue : nothing
      getfield(reg, Symbol(regEntry)) ? error("Regularization $regularizer not supported by solver $solver.") : nothing
    end
    return Kaczmarz(A, reg; kargs...)
  elseif solver == "cgnr"
    for regEntry in RegularizationList()
      regEntry == "L2" ? continue : nothing
      getfield(reg, Symbol(regEntry)) ? error("Regularization $regularizer not supported by solver $solver.") : nothing
    end
    return CGNR(A, reg; kargs...)
  elseif solver== "cg"
    return CG(A;kargs...)
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
    return FISTA(A, reg;kargs...)
  elseif solver == "admm"
    return ADMM(A, reg;kargs...)
  else
    error("Solver $solver not found.")
  end
end

end
