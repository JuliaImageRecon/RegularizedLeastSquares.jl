export pseudoinverse, directSolver


### Direct Solver ###

type DirectSolver <: AbstractLinearSolver
  A
  params
end

DirectSolver(A; kargs...) = DirectSolver(A,kargs)

function init(solver::DirectSolver)
  nothing
end

function deinit(solver::DirectSolver)
  nothing
end

function solve(solver::DirectSolver, u::Vector)
  return directSolver(solver.A, u; solver.params... )
end

#type for Gauß elimination
type tikhonovLU
  S::Matrix
  LUfact
end

tikhonovLU(S::AbstractMatrix) = tikhonovLU(S, lufact(S'*S))

size(A::tikhonovLU) = size(A.S)
length(A::tikhonovLU) = length(A.S)

@doc "This function can be used to calculate the singular values used for Tikhonov regularization." ->
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

type PseudoInverse <: AbstractLinearSolver
  A
  params
end

PseudoInverse(A; kargs...) = PseudoInverse(A,kargs)

function init(solver::PseudoInverse)
  nothing
end

function deinit(solver::PseudoInverse)
  nothing
end

function solve(solver::PseudoInverse, u::Vector)
  return pseudoinverse(solver.A, u; solver.params... )
end


##### SVD #######

#type for singular value decomposition
@doc "This Type stores the singular value decomposition of a Matrix" ->
type SVD
  U::Matrix
  Σ::Vector
  V::Matrix
  D::Vector
end

SVD(U::Matrix,Σ::Vector,V::Matrix) = SVD(U,Σ,V,1./Σ)

size(A::SVD) = (size(A.U,1),size(A.V,1))
length(A::SVD) = prod(size(A))

@doc "This function can be used to calculate the singular values used for Tikhonov regularization." ->
function setlambda(A::SVD, λ::Real)
  for i=1:length(A.Σ)
    σi = A.Σ[i]
    A.D[i] = σi/(σi*σi+λ*λ)
  end
  return nothing
end

# Inversion by using the pseudoinverse of the SVD
@doc "This solves the Tikhonov regularized problem using the singular value decomposition." ->
function pseudoinverse{T}(A::SVD, b::Vector{T}; enforceReal=false, enforcePositive=false, kargs...)
  tmp = BLAS.gemv('C', one(T), A.U, b)
  tmp .*=  A.D
  c = BLAS.gemv('N', one(T), A.V, tmp)

  enforceReal && enfReal!(c)
  enforcePositive && enfPos!(c)

  return c
end
