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
    @inbounds tmp += A.nzval[n]*x[A.rowval[n]]
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

### im2col / col2im ###

@doc "This function rearranges distinct image blocks into columns of a matrix." ->
function im2colDistinct{T}(A::Array{T}, blocksize::NTuple{2,Int64})

  nrows = blocksize[1]
  ncols = blocksize[2]
  nelem = nrows*ncols

  # padding for A such that patches can be formed
  row_ext = mod(size(A,1),nrows)
  col_ext = mod(size(A,2),nrows)
  pad_row = (row_ext != 0)*(nrows-row_ext)
  pad_col = (col_ext != 0)*(ncols-col_ext)

  # rearrange matrix
  A1 = zeros(T, size(A,1)+pad_row, size(A,2)+pad_col)
  A1[1:size(A,1),1:size(A,2)] = A

  t1 = reshape( A1,nrows, floor(Int,size(A1,1)/nrows), size(A,2) )
  t2 = reshape( permutedims(t1,[1 3 2]), size(t1,1)*size(t1,3), size(t1,2) )
  t3 = permutedims( reshape( t2, nelem, floor(Int,size(t2,1)/nelem), size(t2,2) ),[1 3 2] )
  res = reshape(t3,nelem,size(t3,2)*size(t3,3))

  return res
end


@doc "This funtion rearrange columns of a matrix into blocks of an image." ->
function col2imDistinct{T}(A::Array{T}, blocksize::NTuple{2,Int64}, matsize::NTuple{2,Int64})
  # size(A) should not be larger then (blocksize[1]*blocksize[2], matsize[1]*matsize[2]).
  # otherwise the bottom (right) lines (columns) will be cut.
  # matsize should be divisble by blocksize.

  if mod(matsize[1],blocksize[1]) != 0 || mod(matsize[2],blocksize[2]) != 0
    error("matsize should be divisible by blocksize")
  end

  blockrows = blocksize[1]
  blockcols = blocksize[2]
  matrows = matsize[1]
  matcols = matsize[2]
  mblock = floor(Int,matrows/blockrows) # number of blocks per row
  nblock = floor(Int,matcols/blockcols) # number of blocks per column

  # padding for A such that patches can be formed and arranged into a matrix of
  # adequate size
  row_ext = blockrows*blockcols-size(A,1)
  col_ext = mblock*nblock-size(A,2)
  pad_row = (row_ext > 0 )*row_ext
  pad_col = (col_ext > 0 )*col_ext

  A1 = zeros(T, size(A,1)+pad_row, size(A,2)+pad_col)
  A1[1:blockrows*blockcols, 1:mblock*nblock] = A[1:blockrows*blockcols, 1:mblock*nblock]

  # rearrange matrix
  t1 = reshape( A1, blockrows,blockcols,mblock*nblock )
  t2 = reshape( permutedims(t1,[1 3 2]), matrows,nblock,blockcols )
  res = reshape( permutedims(t2,[1 3 2]), matrows,matcols)

end
