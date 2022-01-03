export proxLLR!, normLLR

"""
    proxLLR!(x::Vector{T}, λ::Float64=1e-6; kargs...) where T

proximal map for LLR regularization using singular-value-thresholding

# Arguments
* `x::Vector{T}`                - Vector to apply proximal map to
* `λ::Float64`                  - regularization parameter
* `shape::Tuple{Int}=[]`        - dimensions of the image
* `blockSize::Tuple{Int}=[2;2]` - size of patches to perform singluar value thresholding on
* `randshift::Bool=true`        - randomly shifts the patches to ensure translation invariance
"""
function proxLLR!(x::Vector{T}, λ; shape::NTuple{N,TI}=error(),
   blockSize::NTuple{N,TI}=ntuple(_-> 2, N), randshift::Bool=true) where {T, N,TI <: Integer}

  x = reshape(x, tuple(shape..., length(x) ÷ prod(shape)))

  block_idx = CartesianIndices(blockSize)
  K = size(x)[end]

  if randshift
    # Random.seed!(1234)
    shift_idx = (Tuple(rand(block_idx))..., 0)
    x = circshift(x, shift_idx)
  end

  ext = mod.(shape,blockSize)
  pad = mod.(blockSize .- ext, blockSize)
  if any(pad .!= 0)
    x1 = zeros(T, (shape .+ pad)..., K)
    x1[CartesianIndices(x)] .= x
  else
    x1 = x
  end

  xᴸᴸᴿ = Array{T}(undef, prod(blockSize), K)
  for i ∈ CartesianIndices(StepRange.(0, blockSize, shape .- 1))
    @views xᴸᴸᴿ .= reshape(x1[i .+ block_idx,:], :, K)
    # threshold singular values
    SVDec = svd!(xᴸᴸᴿ)
    proxL1!(SVDec.S,λ)
    x1[i .+ block_idx,:] .= reshape(SVDec.U * Diagonal(SVDec.S) * SVDec.Vt, blockSize..., :)
  end

  if any(pad .!= 0)
    x = x1[CartesianIndices(x)]
  end

  if randshift
    x = circshift(x, -1 .* shift_idx)
  end

  x = vec(x)
  return x
end

"""
    normLLR(x::Vector{T}, λ::Float64; kargs...) where T

returns the value of the LLR-regularization term.
Arguments are the same is in `proxLLR!`
"""
function normLLR(x::Vector{T}, λ::Float64; shape::NTuple{N,TI}, L=1, blockSize::NTuple{N,TI}=ntuple(_-> 2, N), randshift::Bool=true, kargs...) where {N, T, TI <: Integer}

  Nvoxel = prod(shape)
  K = floor(Int,length(x)/(Nvoxel*L))
  normᴸᴸᴿ = 0.
  for i = 1:L
    normᴸᴸᴿ +=  blockNuclearNorm(x[(i-1)*Nvoxel*K+1:i*Nvoxel*K], shape; blockSize=blockSize, randshift=randshift, kargs...)
  end

  return λ*normᴸᴸᴿ
end

function blockNuclearNorm(x::Vector{T}, shape::NTuple{N,TI}; blockSize::NTuple{N,TI}=ntuple(_-> 2, N),
      randshift::Bool=true, kargs...) where {N, T, TI <: Integer}
    x = reshape( x, tuple( shape...,floor(Int64, length(x)/prod(shape)) ) )

    Wy = blockSize[1]
    Wz = blockSize[2]

    if randshift
      srand(1234)
      shift_idx = [rand(1:Wy) rand(1:Wz) 0]
      x = circshift(x, shift_idx)
    end

    ny, nz, K = size(x)

    # reshape into patches
    L = floor(Int,ny*nz/Wy/Wz) # number of patches, assumes that image dimensions are divisble by the blocksizes

    xᴸᴸᴿ = zeros(T,Wy*Wz,L,K)
    for i=1:K
      xᴸᴸᴿ[:,:,i] = im2colDistinct(x[:,:,i], (Wy,Wz))
    end
    xᴸᴸᴿ = permutedims(xᴸᴸᴿ,[1 3 2])

    # L1-norm of singular values
    normᴸᴸᴿ = 0.
    for i = 1:L
      SVDec = svd(xᴸᴸᴿ[:,:,i])
      normᴸᴸᴿ += norm(SVDec.S,1)
    end

    return normᴸᴸᴿ
end
