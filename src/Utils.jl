export SolverInfo, rownorm²

@doc """
The Solver Info type is used to obtain aditional information
of linear solvers and their iteration process
""" ->
type SolverInfo
  resNorm::Vector{Float64}
  xNorm::Vector{Float64}

  SolverInfo() = new( Array(Float64,0), Array(Float64,0))
end

function storeInfo(solverinfo::Void,res,x)
  return nothing
end

function storeInfo(solverinfo::SolverInfo,res,x)
  push!( solverinfo.xNorm, x)
  push!( solverinfo.resNorm, res)
  return nothing
end

### applying sparse trafo ###

A_mul_B!(sparseTrafo::Void, cl) = nothing
At_mul_B!(sparseTrafo::Void, cl) = nothing

### rownorm² ###

@doc "This function computes the 2-norm² of a rows of S for dense matrices." ->
function rownorm²{T,S<:DenseMatrix}(B::MatrixTranspose{T,S},row::Int)
  A = B.data
  U = typeof(real(A[1]))
  res::U = BLAS.nrm2(size(A,1), pointer(A,sub2ind(size(A),1,row)), 1)^2
  return res
end

function rownorm²(A::AbstractMatrix,row::Int)
  T = typeof(real(A[1]))
  res = zero(T)
  @simd for n=1:size(A,2)
    res += abs2(A[row,n])
  end
  return res
end

@doc "This function computes the 2-norm² of a rows of S for dense matrices." ->
function rownorm²{T,S<:SparseMatrixCSC}(B::MatrixTranspose{T,S},row::Int)
  A = B.data
  U = typeof(real(A[1]))
  res::U = BLAS.nrm2(A.colptr[row+1]-A.colptr[row], pointer(A.nzval,A.colptr[row]), 1)^2
  return res
end




### dot_with_matrix_row ###

@doc "This funtion calculates ∑ᵢ Aᵢₖxᵢ for dense matrices." ->
# Fallback implementation
#=function dot_with_matrix_row_simd{T<:Complex}(A::AbstractMatrix{T}, x::Vector{T}, k::Int64)
  res = zero(T)
  @simd for n=1:size(A,2)
    @inbounds res += conj(A[k,n])*x[n]
  end
  return res
end=#

function dot_with_matrix_row{T<:Complex}(A::DenseMatrix{T}, x::Vector{T}, k::Int64)
  BLAS.dotu(length(x), pointer(A,sub2ind(size(A),k,1)), size(A,1), pointer(x,1), 1)
end

function dot_with_matrix_row{T<:Complex,S<:DenseMatrix}(B::MatrixTranspose{T,S}, x::Vector{T}, k::Int64)
  A = B.data
  BLAS.dotu(length(x), pointer(A,sub2ind(size(A),1,k)), 1, pointer(x,1), 1)
end


@doc "This funtion calculates ∑ᵢ Aᵢₖxᵢ for dense matrices." ->
function dot_with_matrix_row{T<:Real}(A::DenseMatrix{T}, x::Vector{T}, k::Int64)
  BLAS.dot(length(x), pointer(A,sub2ind(size(A),k,1)), size(A,1), pointer(x,1), 1)
end

@doc "This funtion calculates ∑ᵢ Aᵢₖxᵢ for dense matrices." ->
function dot_with_matrix_row{T<:Real,S<:DenseMatrix}(B::MatrixTranspose{T,S}, x::Vector{T}, k::Int64)
  A = B.data
  BLAS.dot(length(x), pointer(A,sub2ind(size(A),1,k)), 1, pointer(x,1), 1)
end


@doc "This funtion calculates ∑ᵢ Aᵢₖxᵢ for sparse matrices." ->
function dot_with_matrix_row{T,S<:SparseMatrixCSC}(B::MatrixTranspose{T,S}, x::Vector{T}, k::Int64)
  A = B.data
  tmp = zero(T)
  N = A.colptr[k+1]-A.colptr[k]
  for n=A.colptr[k]:N-1+A.colptr[k]
    @inbounds tmp += conj(A.nzval[n])*x[A.rowval[n]]
  end
  tmp
end




### enfReal! / enfPos! ###

@doc "This funtion enforces the constraint of a real solution." ->
function enfReal!{T<:Complex}(x::Vector{T})
  #Returns x as complex vector with imaginary part set to zero
  @simd for i in 1:length(x)
    @inbounds x[i] = complex(x[i].re)
  end
end

@doc "This funtion enforces the constraint of a real solution." ->
enfReal!{T<:Real}(x::Vector{T}) = nothing

@doc "This funtion enforces positivity constraints on its input." ->
function enfPos!{T<:Complex}(x::Vector{T})
  #Return x as complex vector with negative parts projected onto 0
  @simd for i in 1:length(x)
    @inbounds x[i].re < 0 && (x[i] = im*x[i].im)
  end
end

@doc "This funtion enforces positivity constraints on its input." ->
function enfPos!{T<:Real}(x::Vector{T})
  #Return x as complex vector with negative parts projected onto 0
  @simd for i in 1:length(x)
    @inbounds x[i] < 0 && (x[i] = zero(T))
  end
end


### soft threashold ###

export softThreshold, groupSoftThreshold, svt_distinct, svt_sliding


function softThresholdComplex(z,T)
  #handle the size
  if T>0
    size_z=size(z)
    z=vec(z)

    # This soft thresholding function supports complex numbers
    sz = max(abs(z)-T,0)./(max(abs(z)-T,0)+T).*z

    # Handle the size
    sz=reshape(sz,size_z)
  else
    sz = z;
  end
  return sz
end


#
# soft-thresholding for the Lasso problem
#
function softThreshold(x, lambd,slices::Int64=1)
  soft = copy(x)
  idx = [abs(x[n]) > lambd for n=1:length(x)]
  idxN = [abs(x[n]) <= lambd for n=1:length(x)]
  soft[idxN] = 0.0
  soft[idx] .*= (abs(x[idx]) - lambd) ./ abs(x[idx])
  return soft
end

#
# group-soft-thresholding for l1/l2-regularization
#
function groupSoftThreshold(x, lambd,slices::Int64=1)
  sliceSize = floor(Int,length(x)/slices)
  soft = reshape(x,sliceSize,slices)
  idx = [norm(soft[n,:]) > lambd for n=1:sliceSize]
  idxN = [norm(soft[n,:]) <= lambd for n=1:sliceSize]
  soft[idxN,:] = 0
  xNorm = [norm(soft[n,:]) for n=1:sliceSize]
  soft[idx,:] = soft[idx,:] .* (xNorm[idx] - lambd) ./ xNorm[idx]
  return vec(soft)
end
