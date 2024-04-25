export rownorm², nrmsd

"""
This function computes the 2-norm² of a rows of S for dense matrices.
"""
function rownorm²(B::Transpose{T,S},row::Int64) where {T,S<:DenseMatrix}
  A = B.parent
  U = real(eltype(A))
  res::U = BLAS.nrm2(size(A,1), pointer(A,(LinearIndices(size(A)))[1,row]), 1)^2
  return res
end

function rownorm²(A::AbstractMatrix,row::Int64)
  T = real(eltype(A))
  res = zero(T)
  @simd for n=1:size(A,2)
    res += abs2(A[row,n])
  end
  return res
end

rownorm²(A::AbstractLinearOperator,row::Int64) = rownorm²(Matrix(A[row, :]), 1)
rownorm²(A::ProdOp{T, <:WeightingOp, matT}, row::Int64) where {T, matT} = A.A.weights[row]^2*rownorm²(A.B, row)

"""
This function computes the 2-norm² of a rows of S for sparse matrices.
"""
function rownorm²(B::Transpose{T,S},row::Int64) where {T,S<:SparseMatrixCSC}
  A = B.parent
  U = real(eltype(A))
  res::U = BLAS.nrm2(A.colptr[row+1]-A.colptr[row], pointer(A.nzval,A.colptr[row]), 1)^2
  return res
end

function rownorm²(A, rows)
  res = zero(real(eltype(A)))
  @simd for row in rows
    res += rownorm²(A, row)
  end
  return res
end


### dot_with_matrix_row ###


# Fallback implementation
#=function dot_with_matrix_row_simd{T<:Complex}(A::AbstractMatrix{T}, x::Vector{T}, k::Int64)
  res = zero(T)
  @simd for n=1:size(A,2)
    @inbounds res += conj(A[k,n])*x[n]
  end
  return res
end=#

"""
This funtion calculates ∑ᵢ Aᵢₖxᵢ for dense matrices.
"""
function dot_with_matrix_row(A::DenseMatrix{T}, x::Vector{T}, k::Int64) where {T<:Complex}
  BLAS.dotu(length(x), pointer(A, (LinearIndices(size(A)))[k,1]), size(A,1), pointer(x,1), 1)
end


function dot_with_matrix_row(B::Transpose{T,S},
                       x::Vector{T}, k::Int64) where {T<:Complex,S<:DenseMatrix}
  A = B.parent
  BLAS.dotu(length(x), pointer(A,(LinearIndices(size(A)))[1,k]), 1, pointer(x,1), 1)
end


"""
This funtion calculates ∑ᵢ Aᵢₖxᵢ for dense matrices.
"""
function dot_with_matrix_row(A::DenseMatrix{T}, x::Vector{T}, k::Int64) where {T<:Real}
  BLAS.dot(length(x), pointer(A,(LinearIndices(size(A)))[k,1]), size(A,1), pointer(x,1), 1)
end

"""
This funtion calculates ∑ᵢ Aᵢₖxᵢ for dense matrices.
"""
function dot_with_matrix_row(B::Transpose{T,S},
      x::Vector{T}, k::Int64) where {T<:Real,S<:DenseMatrix}
  A = B.parent
  BLAS.dot(length(x), pointer(A,(LinearIndices(size(A)))[1,k]), 1, pointer(x,1), 1)
end


"""
This funtion calculates ∑ᵢ Aᵢₖxᵢ for sparse matrices.
"""
function dot_with_matrix_row(B::Transpose{T,S},
                             x::Vector{T}, k::Int64) where {T,S<:SparseMatrixCSC}
  A = B.parent
  tmp = zero(T)
  N = A.colptr[k+1]-A.colptr[k]
  for n=A.colptr[k]:N-1+A.colptr[k]
    @inbounds tmp += A.nzval[n]*x[A.rowval[n]]
  end
  tmp
end

function dot_with_matrix_row(prod::ProdOp{T, <:WeightingOp, matT}, x::Vector{T}, k) where {T, matT}
  A = prod.B
  return prod.A.weights[k]*dot_with_matrix_row(A, x, k)
end



### enfReal! / enfPos! ###

"""
This function enforces the constraint of a real solution.
"""
function enfReal!(x::AbstractArray{T}, mask=ones(Bool, length(x))) where {T<:Complex}
  #Returns x as complex vector with imaginary part set to zero
  @simd for i in 1:length(x)
    @inbounds mask[i] && (x[i] = complex(x[i].re))
  end
end

"""
This function enforces the constraint of a real solution.
"""
enfReal!(x::AbstractArray{T}, mask=ones(Bool, length(x))) where {T<:Real} = nothing

"""
This function enforces positivity constraints on its input.
"""
function enfPos!(x::AbstractArray{T}, mask=ones(Bool, length(x))) where {T<:Complex}
  #Return x as complex vector with negative parts projected onto 0
  @simd for i in 1:length(x)
    @inbounds (x[i].re < 0 && mask[i]) && (x[i] = im*x[i].im)
  end
end

"""
This function enforces positivity constraints on its input.
"""
function enfPos!(x::AbstractArray{T}, mask=ones(Bool, length(x))) where {T<:Real}
  #Return x as complex vector with negative parts projected onto 0
  @simd for i in 1:length(x)
    @inbounds (x[i] < 0 && mask[i]) && (x[i] = zero(T))
  end
end

function applyConstraints(x, sparseTrafo, enforceReal, enforcePositive, constraintMask=ones(Bool, length(x)) )

  mask = (constraintMask != nothing) ? constraintMask : ones(Bool, length(x))

  if sparseTrafo != nothing
     x[:] = sparseTrafo * x
  end
  enforceReal && enfReal!(x, mask)
  enforcePositive && enfPos!(x, mask)
  if sparseTrafo != nothing
    x[:] = adjoint(sparseTrafo)*x
  end
end


### im2col / col2im ###

"""
This function rearranges distinct image blocks into columns of a matrix.
"""
function im2colDistinct(A::Array{T}, blocksize::NTuple{2,Int64}) where T

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


"""
This funtion rearrange columns of a matrix into blocks of an image.
"""
function col2imDistinct(A::Array{T}, blocksize::NTuple{2,Int64},
                 matsize::NTuple{2,Int64}) where T
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

### NRMS ###

function nrmsd(I,Ireco)
  N = length(I)

  # This is a little trick. We usually are not interested in simple scalings
  # and therefore "calibrate" them away
  alpha = norm(Ireco)>0 ? (dot(vec(I),vec(Ireco))+dot(vec(Ireco),vec(I))) /
          (2*dot(vec(Ireco),vec(Ireco))) : 1.0
  I2 = Ireco.*alpha

  RMS =  1.0/sqrt(N)*norm(vec(I)-vec(I2))
  NRMS = RMS/(maximum(abs.(I))-minimum(abs.(I)) )
  return NRMS
end

"""
    power_iterations(AᴴA; rtol=1e-2, maxiter=30, verbose=false)

Power iterations to determine the maximum eigenvalue of a normal operator or square matrix.

# Arguments
* `AᴴA`                 - operator or matrix; has to be square

# Keyword Arguments
* `rtol=1e-2`           - relative tolerance; function terminates if the change of the max. eigenvalue is smaller than this values
* `maxiter=30`          - maximum number of power iterations
* `verbose=false`       - print maximum eigenvalue if `true`

# Output
maximum eigenvalue of the operator
"""
function power_iterations(AᴴA; rtol=1e-2, maxiter=30, verbose=false)
  # Creating b like this allows instead of directly randn it to become a CuArray
  b = similar(LinearOperators.storage_type(AᴴA), size(AᴴA, 2))
  b[:] = randn(eltype(AᴴA), size(AᴴA, 2))
  
  bᵒˡᵈ = similar(b)
  λ = Inf

  for i = 1:maxiter
    b ./= norm(b)

    # swap b and bᵒˡᵈ (pointers only, no data is moved or allocated)
    bᵗᵐᵖ = bᵒˡᵈ
    bᵒˡᵈ = b
    b = bᵗᵐᵖ

    mul!(b, AᴴA, bᵒˡᵈ)

    λᵒˡᵈ = λ
    λ = (bᵒˡᵈ' * b) / (bᵒˡᵈ' * bᵒˡᵈ)
    verbose && println("iter = $i; λ = $λ")
    abs(λ/λᵒˡᵈ - 1) < rtol && return λ
  end

  return λ
end