export pseudoinverse, directSolver


### Direct Solver ###

mutable struct DirectSolver <: AbstractLinearSolver
  A
  params
end

DirectSolver(A; kargs...) = DirectSolver(A,kargs)

function solve(solver::DirectSolver, u::Vector)
  return directSolver(solver.A, u; solver.params... )
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

mutable struct PseudoInverse <: AbstractLinearSolver
  A
  params
end

PseudoInverse(A; kargs...) = PseudoInverse(A,kargs)

function solve(solver::PseudoInverse, u::Vector)
  return pseudoinverse(solver.A, u; solver.params... )
end


##### SVD #######

#type for singular value decomposition
"""
This Type stores the singular value decomposition of a Matrix
"""
mutable struct SVD
  U::Matrix
  Σ::Vector
  V::Matrix
  D::Vector
end

SVD(U::Matrix,Σ::Vector,V::Matrix) = SVD(U, Σ, V, 1. / Σ)

Base.size(A::SVD) = (size(A.U,1),size(A.V,1))
Base.length(A::SVD) = prod(size(A))

"""
This function can be used to calculate the singular values used for
Tikhonov regularization.
"""
function setlambda(A::SVD, λ::Real)
  for i=1:length(A.Σ)
    σi = A.Σ[i]
    A.D[i] = σi/(σi*σi+λ*λ)
  end
  return nothing
end

# Inversion by using the pseudoinverse of the SVD
"""
This solves the Tikhonov regularized problem using the singular value decomposition.
"""
function pseudoinverse(A::SVD, b::Vector{T}; enforceReal=false,
                       enforcePositive=false, kargs...) where T
  tmp = BLAS.gemv('C', one(T), A.U, b)
  tmp .*=  A.D
  c = BLAS.gemv('N', one(T), A.V, tmp)

  enforceReal && enfReal!(c)
  enforcePositive && enfPos!(c)

  return c
end
