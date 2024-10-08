export PseudoInverse, DirectSolver
### Direct Solver ###

mutable struct DirectSolver{matT, R, PR}  <: AbstractDirectSolver
  A::matT
  l2::R
  normalizeReg::AbstractRegularizationNormalization
  proj::Vector{PR}
  state::AbstractSolverState{<:AbstractDirectSolver}
end

mutable struct DirectSolverState{vecT} <: AbstractSolverState{DirectSolver}
  x::vecT
  b::vecT
end

function DirectSolver(A; reg::Vector{<:AbstractRegularization} = [L2Regularization(zero(real(eltype(A))))], normalizeReg::AbstractRegularizationNormalization = NoNormalization())
  reg = normalize(DirectSolver, normalizeReg, reg, A, nothing)
  idx = findsink(L2Regularization, reg)
  if isnothing(idx)
    L2 = L2Regularization(zero(T))
  else
    L2 = reg[idx]
    deleteat!(reg, idx)
  end

  indices = findsinks(AbstractProjectionRegularization, reg)
  other = AbstractRegularization[reg[i] for i in indices]
  deleteat!(reg, indices)
  if length(reg) == 1
    push!(other, reg[1])
  elseif length(reg) > 1
    error("PseudoInverse does not allow for more than one L2 regularization term, found $(length(reg))")
  end
  other = identity.(other)

  T = eltype(A)
  x = Vector{T}(undef,size(A, 2))
  b = zeros(T, size(A,1))

  return DirectSolver(A, L2, normalizeReg, other, DirectSolverState(x, b))
end

function init!(solver::DirectSolver, state::DirectSolverState{vecT}, b::otherT; kwargs...) where {vecT, otherT <: AbstractVector}
  x = similar(b, size(state.x)...)
  bvecT = similar(b, size(state.b)...)
  solver.state = DirectSolverState(x, bvecT)
  init!(solver, solver.state, b; kwargs...)
end
function init!(solver::DirectSolver, state::DirectSolverState{vecT}, b::vecT; x0=0) where vecT <: AbstractVector
  solver.l2 = normalize(solver, solver.normalizeReg, solver.l2, solver.A, b)
  state.b .= b
  state.x .= x0
end

function iterate(solver::DirectSolver, state::DirectSolverState)
  A = solver.A
  λ_ = λ(solver.l2)
  lufact = lu(A'*A .+ λ_)
  x = \(lufact,A' * state.b)

  for p in solver.proj
    prox!(p, x)
  end
  state.x .= x
  return nothing
end

#type for Gauß elimination
mutable struct tikhonovLU
  S::Matrix
  LUfact
end

tikhonovLU(S::AbstractMatrix) = tikhonovLU(S, lufact(S'*S))

Base.size(A::tikhonovLU) = size(A.S)
Base.length(A::tikhonovLU) = length(A.S)

"""
This function can be used to calculate the singular values used for Tikhonov regularization.
"""
function setlambda(A::tikhonovLU, λ::Real)
  A.LUfact = lufact(A.S'*A.S + λ*speye(size(A,2),size(A,2)))
  return nothing
end


# Simple Tikonov regularized reconstruction
function directSolver(A::tikhonovLU, b::Vector; enforceReal=false, enforcePositive=false, kargs...)
  x = \(A.LUfact,A.S' * b)

  enforceReal ? enfReal!(x) : nothing
  enforcePositive ? enfPos!(x) : nothing

  return x
end

###  Pseudoinverse ###

mutable struct PseudoInverse{R, PR}  <: AbstractDirectSolver
  svd::SVD
  l2::R
  normalizeReg::AbstractRegularizationNormalization
  proj::Vector{PR}
  state::AbstractSolverState{<:AbstractDirectSolver}
end

function PseudoInverse(A; reg::Vector{<:AbstractRegularization} = [L2Regularization(zero(real(eltype(A))))], normalizeReg::AbstractRegularizationNormalization = NoNormalization())
  reg = normalize(PseudoInverse, normalizeReg, reg, A, nothing)
  idx = findsink(L2Regularization, reg)
  if isnothing(idx)
    L2 = L2Regularization(zero(T))
  else
    L2 = reg[idx]
    deleteat!(reg, idx)
  end

  indices = findsinks(AbstractProjectionRegularization, reg)
  other = AbstractRegularization[reg[i] for i in indices]
  deleteat!(reg, indices)
  if length(reg) == 1
    push!(other, reg[1])
  elseif length(reg) > 1
    error("PseudoInverse does not allow for more than one L2 regularization term, found $(length(reg))")
  end
  other = identity.(other)

  T = eltype(A)
  x = Vector{T}(undef,size(A, 2))
  b = zeros(T, size(A,1))

  return PseudoInverse(A, x, b, L2, normalizeReg, other)
end
function PseudoInverse(A::AbstractMatrix, x, b, l2, norm, proj)
  u, s, v = svd(A)
  temp = SVD(u, s, v)
  return PseudoInverse(temp, l2, norm, proj, DirectSolverState(x, b))
end

function init!(solver::PseudoInverse, state::DirectSolverState{vecT}, b::otherT; kwargs...) where {vecT, otherT <: AbstractVector}
  x = similar(b, size(state.x)...)
  bvecT = similar(b, size(state.b)...)
  solver.state = DirectSolverState(x, bvecT)
  init!(solver, solver.state, b; kwargs...)
end
function init!(solver::PseudoInverse, state::DirectSolverState{vecT}, b::vecT; x0=0) where vecT <: AbstractVector
  solver.l2 = normalize(solver, solver.normalizeReg, solver.l2, solver.svd, b)
  state.b .= b
end

function iterate(solver::PseudoInverse, state::DirectSolverState)
  # Inversion by using the pseudoinverse of the SVD
  svd = solver.svd

  # Calculate singular values used for tikhonov regularization
  λ_ = λ(solver.l2)
  D = svd.S ./ (svd.S.*svd.S .+ λ_ )

  tmp = adjoint(svd.U)*state.b
  tmp .*= D
  x = svd.Vt * tmp

  for p in solver.proj
    prox!(p, x)
  end
  state.x = x
  return nothing
end