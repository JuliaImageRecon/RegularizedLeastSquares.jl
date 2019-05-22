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
function proxLLR!(x::Vector{T}, λ::Float64=1e-6; shape::NTuple=[],
   blockSize::Array{Int64,1}=[2; 2], randshift::Bool=true, kargs...) where T
  # xᵖʳᵒˣ = zeros(T,size(x))
  N = prod(shape)
  # K = floor(Int,length(x)/N)
  x[:] = vec( svt(x[:], shape, λ; blockSize=blockSize, randshift=randshift, kargs...) )

  return x
end

function svt(x::Vector{T}, shape::Tuple, λ::Float64=1e-6;
   blockSize::Array{Int64,1}=[2; 2], randshift::Bool=true, kargs...) where T

  x = reshape( x, tuple( shape...,floor(Int64, length(x)/prod(shape)) ) )

  Wy = blockSize[1]
  Wz = blockSize[2]

  if randshift
    # Random.seed!(1234)
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

  # threshold singular values
  for i = 1:L
    if xᴸᴸᴿ[:,:,i] == zeros(T, Wy*Wz,K)
      continue
    end
    SVDec = svd(xᴸᴸᴿ[:,:,i])
    proxL1!(SVDec.S,λ)
    xᴸᴸᴿ[:,:,i] = SVDec.U*Matrix(Diagonal(SVDec.S))*SVDec.Vt
  end

  # reshape into image
  xᵗʰʳᵉˢʰ = zeros(T,size(x))
  for i = 1:K
    xᵗʰʳᵉˢʰ[:,:,i] = col2imDistinct( xᴸᴸᴿ[:,i,:], (Wy,Wz), (ny,nz) )
  end

  if randshift
    xᵗʰʳᵉˢʰ = circshift(xᵗʰʳᵉˢʰ, -1*shift_idx)
  end

  if !isempty(shape)
    xᵗʰʳᵉˢʰ = reshape( xᵗʰʳᵉˢʰ, prod(shape),floor( Int, length(xᵗʰʳᵉˢʰ)/prod(shape) ) )
  end

  return xᵗʰʳᵉˢʰ
end

"""
    normLLR(x::Vector{T}, λ::Float64; kargs...) where T

returns the value of the LLR-regularization term.
Arguments are the same is in `proxLLR!`
"""
function normLLR(x::Vector{T}, λ::Float64; shape::NTuple=[], L=1, blockSize::Array{Int64,1}=[2; 2], randshift::Bool=true, kargs...) where T

  N = prod(shape)
  K = floor(Int,length(x)/(N*L))
  normᴸᴸᴿ = 0.
  for i = 1:L
    normᴸᴸᴿ +=  blockNuclearNorm(x[(i-1)*N*K+1:i*N*K], shape; blockSize=blockSize, randshift=randshift, kargs...)
  end

  return λ*normᴸᴸᴿ
end

function blockNuclearNorm(x::Vector{T}, shape::Tuple; blockSize::Array{Int64,1}=[2; 2],
      randshift::Bool=true, kargs...) where T
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
