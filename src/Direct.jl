export PseudoInverse, DirectSolver


### Direct Solver ###

mutable struct DirectSolver{matT, R, PR}  <: AbstractDirectSolver
  A::matT
  l2::R
  normalizeReg::AbstractRegularizationNormalization
  proj::Vector{PR}
end

function DirectSolver(A; reg::Vector{<:AbstractRegularization} = [L2Regularization(zero(real(eltype(A))))], normalizeReg::AbstractRegularizationNormalization = NoNormalization(), kargs...)
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

  return DirectSolver(A, L2, normalizeReg, other)
end

function solve!(solver::DirectSolver, b::Vector)
  solver.l2 = normalize(solver, solver.normalizeReg, solver.l2, solver.A, b)

  A = solver.A
  λ_ = λ(solver.l2)
  lufact = lu(Matrix(A'*A + λ_*opEye(size(A,2),size(A,2))))
  x = \(lufact,A' * b)

  for p in solver.proj
    prox!(p, x)
  end
  return x
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
end

function PseudoInverse(A; reg::Vector{<:AbstractRegularization} = [L2Regularization(zero(real(eltype(A))))], normalizeReg::AbstractRegularizationNormalization = NoNormalization(), kargs...)
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

  return PseudoInverse(A, L2, normalizeReg, other)
end
function PseudoInverse(A::AbstractMatrix, l2, norm, proj)
  u, s, v = svd(A)
  temp = SVD(u, s, v)
  return PseudoInverse(temp, l2, norm, proj)
end

function solve!(solver::PseudoInverse, b::Vector{T}) where T
  solver.l2 = normalize(solver, solver.normalizeReg, solver.l2, solver.svd, b)

  # Inversion by using the pseudoinverse of the SVD
  svd = solver.svd

  # Calculate singular values used for tikhonov regularization
  D = [1/s for s in svd.S]
  λ_ = λ(solver.l2)
  for i=1:length(D)
    σi = svd.S[i]
    D[i] = σi/(σi*σi+λ_*λ_)
  end

  tmp = BLAS.gemv('C', one(T), svd.U, b)
  tmp .*=  D
  c = BLAS.gemv('N', one(T), svd.Vt, tmp)

  for p in solver.proj
    prox!(p, c)
  end
  return c
end