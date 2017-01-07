export kaczmarz

type Kaczmarz <: AbstractLinearSolver
  A
  params
end

Kaczmarz(A; kargs...) = Kaczmarz(A,kargs)

function solve(solver::Kaczmarz, u::Vector)
  return kaczmarz(solver.A, u; solver.params... )
end

### initkaczmarzfast ###

@doc "This funtion saves the denominators to compute αl in denom and the rowindices, which lead to an update of cl in rowindex." ->
function initkaczmarzfast(S::AbstractMatrix,λ,weights::Vector)
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

### kaczmarz_update_simd! ###

@doc "This funtion updates x during the kaczmarz algorithm for dense matrices." ->
function kaczmarz_update_simd!{T}(A::DenseMatrix{T}, x::Vector, k::Integer, beta)
  @simd for n=1:size(A,2)
    @inbounds x[n] += beta*conj(A[k,n])
  end
end

@doc "This funtion updates x during the kaczmarz algorithm for dense matrices." ->
function kaczmarz_update_simd!{T,S<:DenseMatrix}(B::MatrixTranspose{T,S}, x::Vector, k::Integer, beta)
  A = B.data
  @simd for n=1:size(A,1)
    @inbounds x[n] += beta*conj(A[n,k])
  end
end

#=
@doc "This funtion updates x during the kaczmarz algorithm for dense matrices." ->
function kaczmarz_update_simd!{T}(A::Matrix{T}, x::Vector{T}, k::Integer, beta::T)
  BLAS.axpy!(length(x), beta, pointer(A,sub2ind(size(A),1,k)), 1, pointer(x), 1)
end
=#

@doc "This funtion updates x during the kaczmarz algorithm for sparse matrices." ->
function kaczmarz_update_simd!{T,S<:SparseMatrixCSC}(B::MatrixTranspose{T,S}, x::Vector, k::Integer, beta)
  A = B.data
  N = A.colptr[k+1]-A.colptr[k]
  for n=A.colptr[k]:N-1+A.colptr[k]
    @inbounds x[A.rowval[n]] += beta*conj(A.nzval[n])
  end
end

### kaczmarz ###

@doc """
This funtion implements the kaczmarz algorithm.

### Keyword/Optional Arguments

* `lambd::Float64`: The regularization parameter, relative to the matrix trace
* `iterations::Int`: Number of iterations of the iterative solver
* `solver::AbstractString`: Algorithm used to solve the imaging equation (currently "kaczmarz" or "cgnr")
* `normWeights::Bool`: Enable row normalization (true/false)
* `sparseTrafo::AbstractString`: Enable sparseTrafo if set to "DCT" or "FFT".
* `shuff::Bool`: Enable shuffeling of rows during iterations in the kaczmarz algorithm.
* `enforceReal::Bool`: Enable projection of solution on real plane during iteration.
* `enforcePositive::Bool`: Enable projection of solution onto positive halfplane during iteration.
""" ->
function kaczmarz(S, u::Vector;
 iterations=10, lambd=0.0, weights=nothing, enforceReal=false, enforcePositive=false, sparseTrafo=nothing, startVector=nothing, solverInfo=nothing, verbose = true ,kargs...)
  T = typeof(real(u[1]))
  lambd = convert(T,lambd)
  weights==nothing ? weights=ones(T,size(S,1)) : nothing #search for positive solution as default
  startVector==nothing ? startVector=zeros(typeof(u[1]),size(S,2)) : nothing
  return kaczmarz(S, u, startVector, iterations, lambd, weights, enforceReal, enforcePositive, sparseTrafo, solverInfo, verbose=verbose)
end

function kaczmarz{T}(S, u::Vector{T}, startVector, iterations, lambd, weights, enforceReal, enforcePositive, sparseTrafo, solverInfo; verbose = true)
  # fast implementation of kaczmarz using SIMD instructions
  M::Int64 = size(S,1)      #number of rows of system matrix
  N::Int64 = size(S,2)      #number of cols of system matrix
  denom, rowindex = initkaczmarzfast(S,lambd,weights) #denom necessary to update αl, if rownorm ≠ 0. rowindex contains the indeces of nonzero rows.
  rowIndexCycle = collect(1:length(rowindex))

  cl = startVector     #solution vector
  vl = zeros(T,M)     #residual vector

  ɛw = zeros(T,length(rowindex))
  for i=1:length(rowindex)
    j = rowindex[i]
    ɛw[i] = sqrt(lambd)/weights[j]
  end

  #verbose && (p = Progress(iterations, 1, "Kaczmarz Iteration..."))
  for l=1:iterations
    for i in rowIndexCycle
      j = rowindex[i]
      τl = dot_with_matrix_row_simd(S,cl,j)
      αl = denom[i]*(u[j]-τl-ɛw[i]*vl[j])
      kaczmarz_update_simd!(S,cl,j,αl)
      vl[j] += αl*ɛw[i]
    end

    # invoke constraints
    A_mul_B!(sparseTrafo, cl)
    enforceReal && enfReal!(cl)
    enforcePositive && enfPos!(cl)
    At_mul_B!(sparseTrafo, cl)

    solverInfo != nothing && storeInfo(solverInfo,norm(S*cl-u),norm(cl))
    #verbose && next!(p)
  end
  return cl
end
