export SolverInfo, rownorm², nrmsd

"""
The Solver Info type is used to obtain aditional information
of linear solvers and their iteration process
"""
mutable struct SolverInfo
  cost::Vector{Float64}
  residual::Vector
  relSolutionChange::Vector{Float64}
  nrmse::Vector{Float64}
  x_ref::Union{Vector,Nothing}
  iterations::Union{Vector,Nothing}
  x_iter::Union{Vector,Nothing}
end

function SolverInfo(x_ref::Union{Vector,Nothing}=nothing; store_solutions=false,
    kargs...)
  x_iter = store_solutions ? Vector{Any}() : nothing
  SolverInfo(Vector{Float64}(), Vector{Float64}(), Vector{Float64}(),
             Vector{Float64}(), x_ref, Vector{Int64}(), x_iter)
end

function storeInfo(solverinfo::Nothing, res, x)
  return nothing
end

function storeInfo(solverInfo::SolverInfo,A,y::Vector{U},x::Vector{U};
                       xᵒˡᵈ::Vector{U}=U[],reg::Union{Vector{T},Nothing}=nothing,
                       residual::Vector{U}=U[]) where {U,T<:AbstractRegularization}
  # residual
  if isempty(residual)
    residual=A*x-y
  end
  push!(solverInfo.residual,norm(residual))
  # solution
  if solverInfo.x_iter != nothing
    push!(solverInfo.x_iter, deepcopy(x))
  end
  # cost function
  cost = 0.5*norm(residual)^2
  if reg != nothing
    for i=1:length(reg)
      cost += reg[i].norm(x,reg[i].λ;reg[i].params...)
    end
  end
  push!(solverInfo.cost, cost)
  # relative change of the solution
  if !isempty(xᵒˡᵈ)
    relSolutionChange = norm(x-xᵒˡᵈ)/norm(x)
    push!(solverInfo.relSolutionChange, relSolutionChange)
  end
  # nrmse
  if solverInfo.x_ref != nothing
    nrmse = nrmsd(solverInfo.x_ref,x)
    push!(solverInfo.nrmse, nrmse)
  end
end

function storeIter(solverInfo::SolverInfo, iterations::Int64)
  if solverInfo.iterations != nothing
    push!(solverInfo.iterations, iterations)
  end
end

function storeInfo(solverinfo::SolverInfo, res, x)
  push!( solverinfo.xNorm, x)
  push!( solverinfo.resNorm, res)
  return nothing
end

function storeResidual(solverinfo::SolverInfo,res)
  push!( solverinfo.resNorm, res)
  return nothing
end

function storeRegularization(solverinfo::SolverInfo,regNorm)
  push!( solverinfo.xNorm, regNorm)
  return nothing
end

### rownorm² ###

"""
This function computes the 2-norm² of a rows of S for dense matrices.
"""
function rownorm²(B::Transpose{T,S},row::Int) where {T,S<:DenseMatrix}
  A = B.parent
  U = typeof(real(A[1]))
  res::U = BLAS.nrm2(size(A,1), pointer(A,(LinearIndices(size(A)))[1,row]), 1)^2
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

"""
This function computes the 2-norm² of a rows of S for dense matrices.
"""
function rownorm²(B::Transpose{T,S},row::Int) where {T,S<:SparseMatrixCSC}
  A = B.parent
  U = typeof(real(A[1]))
  res::U = BLAS.nrm2(A.colptr[row+1]-A.colptr[row], pointer(A.nzval,A.colptr[row]), 1)^2
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




### enfReal! / enfPos! ###

"""
This function enforces the constraint of a real solution.
"""
function enfReal!(x::Vector{T}, mask=ones(Bool, length(x))) where {T<:Complex}
  #Returns x as complex vector with imaginary part set to zero
  @simd for i in 1:length(x)
    @inbounds mask[i] && (x[i] = complex(x[i].re))
  end
end

"""
This function enforces the constraint of a real solution.
"""
enfReal!(x::Vector{T}, mask=ones(Bool, length(x))) where {T<:Real} = nothing

"""
This function enforces positivity constraints on its input.
"""
function enfPos!(x::Vector{T}, mask=ones(Bool, length(x))) where {T<:Complex}
  #Return x as complex vector with negative parts projected onto 0
  @simd for i in 1:length(x)
    @inbounds (x[i].re < 0 && mask[i]) && (x[i] = im*x[i].im)
  end
end

"""
This function enforces positivity constraints on its input.
"""
function enfPos!(x::Vector{T}, mask=ones(Bool, length(x))) where {T<:Real}
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
    x[:] = sparseTrafo' * x
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
