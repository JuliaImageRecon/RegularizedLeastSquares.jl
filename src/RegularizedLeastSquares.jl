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
using LinearOperatorCollection
using InteractiveUtils

export AbstractLinearSolver, createLinearSolver, init, deinit, solve, linearSolverList, linearSolverListReal, applicableSolverList

abstract type AbstractLinearSolver end

export AbstractRowActionSolver
abstract type AbstractRowActionSolver <: AbstractLinearSolver end

export AbstractDirectSolver
abstract type AbstractDirectSolver <: AbstractLinearSolver end

export AbstractPrimalDualSolver
abstract type AbstractPrimalDualSolver <: AbstractLinearSolver end

export AbstractProximalGradientSolver
abstract type AbstractProximalGradientSolver <: AbstractLinearSolver end

export AbstractKrylovSolver
abstract type AbstractKrylovSolver <: AbstractLinearSolver end

# Fallback function
setlambda(S::AbstractMatrix, λ::Real) = nothing

include("Regularization.jl")
include("proximalMaps/ProxL1.jl")
include("proximalMaps/ProxL2.jl")
include("proximalMaps/ProxL21.jl")
include("proximalMaps/ProxLLR.jl")
# include("proximalMaps/ProxSLR.jl")
include("proximalMaps/ProxPositive.jl")
include("proximalMaps/ProxProj.jl")
include("proximalMaps/ProxReal.jl")
include("proximalMaps/ProxTV.jl")
include("proximalMaps/ProxTVCondat.jl")
include("proximalMaps/ProxNuclear.jl")
include("proximalMaps/ScaledRegularization.jl")
include("proximalMaps/TransformedRegularization.jl")
include("proximalMaps/MaskedRegularization.jl")

include("Utils.jl")
include("Kaczmarz.jl")
include("DAXKaczmarz.jl")
include("DAXConstrained.jl")
include("CGNR.jl")
include("Direct.jl")
include("FISTA.jl")
include("OptISTA.jl")
include("POGM.jl")
include("ADMM.jl")
include("SplitBregman.jl")
include("PrimalDualSolver.jl")

"""
Return a list of all available linear solvers
"""
function linearSolverList()
  filter(s -> s ∉ [DaxKaczmarz, DaxConstrained, PrimalDualSolver], linearSolverListReal())
end

function linearSolverListReal()
  union(subtypes.(subtypes(AbstractLinearSolver))...) # For deeper nested type extend this to loop for types with isabstracttype == true
end

export isapplicable
isapplicable(solver::AbstractLinearSolver, args...) = isapplicable(typeof(solver), args...)
isapplicable(x, reg::AbstractRegularization) = isapplicable(x, [reg])
isapplicable(::Type{T}, reg::Vector{<:AbstractRegularization}) where T <: AbstractLinearSolver = false

function isapplicable(::Type{T}, reg::Vector{<:AbstractRegularization}) where T <: AbstractRowActionSolver
  applicable = true
  applicable &= length(findsinks(AbstractParameterizedRegularization, reg)) <= 2
  applicable &= length(findsinks(L2Regularization, reg)) == 1
  return applicable
end

function isapplicable(::Type{T}, reg::Vector{<:AbstractRegularization}) where T <: AbstractPrimalDualSolver
  # TODO
  return true
end

function isapplicable(::Type{T}, reg::Vector{<:AbstractRegularization}) where T <: AbstractProximalGradientSolver
  applicable = true
  applicable &= length(findsinks(AbstractParameterizedRegularization, reg)) == 1
  return applicable
end

function isapplicable(::Type{T}, A, x) where T <: AbstractLinearSolver
  # TODO
  applicable = true
  return applicable
end

isapplicable(::Type{T}, A, x, reg) where T <: AbstractLinearSolver = isapplicable(T, A, x) && isapplicable(T, reg)

applicableSolverList(args...) = filter(solver -> isapplicable(solver, args...), linearSolverListReal())

"""
    createLinearSolver(solver::AbstractLinearSolver, A; log::Bool=false, kargs...)

This method creates a solver. The supported solvers are methods typically used for solving
regularized linear systems. All solvers return an approximate solution to Ax = b.

TODO: give a hint what solvers are available
"""
function createLinearSolver(solver::Type{T}, A; log::Bool=false, kargs...) where {T<:AbstractLinearSolver}
  log ? solverInfo = SolverInfo(;kargs...) : solverInfo=nothing
  return solver(A; kargs...)
end

function createLinearSolver(solver::Type{T}, A, x; log::Bool=false, kargs...) where {T<:AbstractLinearSolver}
  log ? solverInfo = SolverInfo(;kargs...) : solverInfo=nothing
  return solver(A, x; kargs...)
end


end
