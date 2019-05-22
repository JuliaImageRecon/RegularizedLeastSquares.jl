export kaczmarz

mutable struct Kaczmarz <: AbstractLinearSolver
  A
  reg::Regularization
  params
end

"""
    Kaczmarz(A; λ = 0.0, reg = Regularization("L2", λ), kargs...)

creates an `Kaczmarz` object for the system matrix `A`.

# Arguments
* `A`                             - system matrix
* (`λ=0.0`)                       - Regularization paramter
* (`reg=Regularization("L2", λ)`) - Regularization object
"""
function Kaczmarz(A; λ = 0.0, reg = Regularization("L2", λ), kargs...)
  if (reg.prox!) != (proxL2!)
    @error "Kaczmarz only supports L2 regularizer"
  end
  return Kaczmarz(A,reg,kargs)
end

"""
    solve(solver::Kaczmarz, u::Vector)

solves Thkhonov-regularized inverse problem using Kaczmarz algorithm.

# Arguments
* `solver::Kaczmarz  - the solver containing both system matrix and regularizer
* `u::Vector`        - data vector
"""
function solve(solver::Kaczmarz, u::Vector)
  return kaczmarz(solver.A, u; λ=solver.reg.λ, solver.params... )
end

### initkaczmarz ###

"""
    initkaczmarz(S::AbstractMatrix,λ,weights::Vector)

This funtion saves the denominators to compute αl in denom and the rowindices,
which lead to an update of cl in rowindex.
"""
function initkaczmarz(S::AbstractMatrix,λ,weights::Vector)
  T = typeof(real(S[1]))
  denom = T[]
  rowindex = Int64[]

  for i=1:size(S,1)
    s² = rownorm²(S,i)*weights[i]^2
    if s²>0
      push!(denom,weights[i]^2/(s²+λ))
      push!(rowindex,i)
    end
  end
  denom, rowindex
end

### kaczmarz_update! ###

"""
    kaczmarz_update!(A::DenseMatrix{T}, x::Vector, k::Integer, beta) where T

This funtion updates x during the kaczmarz algorithm for dense matrices.
"""
function kaczmarz_update!(A::DenseMatrix{T}, x::Vector, k::Integer, beta) where T
  @simd for n=1:size(A,2)
    @inbounds x[n] += beta*conj(A[k,n])
  end
end

"""
    kaczmarz_update!(B::Transpose{T,S}, x::Vector,
                          k::Integer, beta) where {T,S<:DenseMatrix}

This funtion updates x during the kaczmarz algorithm for dense matrices.
"""
function kaczmarz_update!(B::Transpose{T,S}, x::Vector,
                          k::Integer, beta) where {T,S<:DenseMatrix}
  A = B.parent
  @simd for n=1:size(A,1)
    @inbounds x[n] += beta*conj(A[n,k])
  end
end

#=
@doc "This funtion updates x during the kaczmarz algorithm for dense matrices." ->
function kaczmarz_update!{T}(A::Matrix{T}, x::Vector{T}, k::Integer, beta::T)
  BLAS.axpy!(length(x), beta, pointer(A,sub2ind(size(A),1,k)), 1, pointer(x), 1)
end
=#

"""
    kaczmarz_update!(B::Transpose{T,S}, x::Vector,
                          k::Integer, beta) where {T,S<:SparseMatrixCSC}

This funtion updates x during the kaczmarz algorithm for sparse matrices.
"""
function kaczmarz_update!(B::Transpose{T,S}, x::Vector,
                          k::Integer, beta) where {T,S<:SparseMatrixCSC}
  A = B.parent
  N = A.colptr[k+1]-A.colptr[k]
  for n=A.colptr[k]:N-1+A.colptr[k]
    @inbounds x[A.rowval[n]] += beta*conj(A.nzval[n])
  end
end

### kaczmarz ###

"""
    kaczmarz(S, u::Vector; kargs...)

This funtion implements the kaczmarz algorithm.

# Keyword/Optional Arguments
* `λ::Float64`: The regularization parameter, relative to the matrix trace
* `iterations::Int`: Number of iterations of the iterative solver
* `solver::AbstractString`: Algorithm used to solve the imaging equation (currently "kaczmarz" or "cgnr")
* `normWeights::Bool`: Enable row normalization (true/false)
* `sparseTrafo::AbstractString`: Enable sparseTrafo if set to "DCT-IV" or "FFT".
* `shuff::Bool`: Enable shuffeling of rows during iterations in the kaczmarz algorithm.
* `enforceReal::Bool`: Enable projection of solution on real plane during iteration.
* `enforcePositive::Bool`: Enable projection of solution onto positive halfplane during iteration.
"""
function kaczmarz(S, u::Vector;
 iterations=10, λ=0.0, weights=nothing, enforceReal=false, shuffleRows=false, enforcePositive=false,
 sparseTrafo=nothing, startVector=nothing, solverInfo=nothing, seed=1234, kargs...)
  T = typeof(real(u[1]))
  λ = convert(T,λ)
  weights==nothing ? weights=ones(T,size(S,1)) : nothing #search for positive solution as default
  startVector==nothing ? startVector=zeros(typeof(u[1]),size(S,2)) : nothing
  return kaczmarz(S, u, startVector, iterations, λ, weights, enforceReal, enforcePositive,
                  sparseTrafo, solverInfo)
end

function kaczmarz(S, u::Vector{T}, startVector, iterations, λ, weights, enforceReal,
            enforcePositive, sparseTrafo, solverInfo, shuffleRows=false, seed=1234) where T
  # fast implementation of kaczmarz using SIMD instructions
  M::Int64 = size(S,1)      #number of rows of system matrix
  N::Int64 = size(S,2)      #number of cols of system matrix
  denom, rowindex = initkaczmarz(S,λ,weights) #denom necessary to update αl, if rownorm ≠ 0. rowindex contains the indeces of nonzero rows.
  rowIndexCycle = collect(1:length(rowindex))

  if shuffleRows
    Random.seed!(seed)
    shuffle!(rowIndexCycle)
  end

  cl = startVector     #solution vector
  vl = zeros(T,M)     #residual vector

  ɛw = zeros(T,length(rowindex))
  for i=1:length(rowindex)
    j = rowindex[i]
    ɛw[i] = sqrt(λ)/weights[j]
  end

  reg = Regularization("L2", λ)

  for l=1:iterations
    for i in rowIndexCycle
      j = rowindex[i]
      τl = dot_with_matrix_row(S,cl,j)
      αl = denom[i]*(u[j]-τl-ɛw[i]*vl[j])
      kaczmarz_update!(S,cl,j,αl)
      vl[j] += αl*ɛw[i]
    end

    # invoke constraints
    applyConstraints(cl, sparseTrafo, enforceReal, enforcePositive)

    # solverInfo != nothing && storeInfo(solverInfo,norm(S*cl-u),norm(cl))
    solverInfo != nothing && storeInfo(solverInfo,S,u,cl;reg=[reg],residual=vl)
  end
  return cl
end
