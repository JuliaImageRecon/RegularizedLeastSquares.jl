export LLRRegularization

"""
    LLRRegularization

Regularization term implementing the proximal map for locally low rank (LLR) regularization using singular-value-thresholding.

# Arguments
* `λ`                  - regularization paramter

# Keywords
* `shape::Tuple{Int}`            - dimensions of the image
* `blockSize::Tuple{Int}=(2,2)`  - size of patches to perform singular value thresholding on
* `randshift::Bool=true`         - randomly shifts the patches to ensure translation invariance
* `fullyOverlapping::Bool=false` - choose between fully overlapping block or non-overlapping blocks
"""
struct LLRRegularization{T, N, TI} <: AbstractParameterizedRegularization{T} where {N, TI<:Integer}
  λ::T
  shape::NTuple{N,TI}
  blockSize::NTuple{N,TI}
  randshift::Bool
  fullyOverlapping::Bool
  L::Int64
end
LLRRegularization(λ;  shape::NTuple{N,TI}, blockSize::NTuple{N,TI} = ntuple(_ -> 2, N), randshift::Bool = true, fullyOverlapping::Bool = false, L::Int64 = 1, kargs...) where {N,TI<:Integer} =
 LLRRegularization(λ, shape, blockSize, randshift, fullyOverlapping, L)

"""
    prox!(reg::LLRRegularization, x, λ)

performs the proximal map for LLR regularization using singular-value-thresholding
"""
function prox!(reg::LLRRegularization, x::Union{AbstractArray{T}, AbstractArray{Complex{T}}}, λ::T) where {T <: Real}
    reg.fullyOverlapping ? proxLLROverlapping!(reg, x, λ) : proxLLRNonOverlapping!(reg, x, λ)
end

"""
    proxLLRNonOverlapping!(reg::LLRRegularization, x, λ)

performs the proximal map for LLR regularization using singular-value-thresholding on non-overlapping blocks
"""
function proxLLRNonOverlapping!(reg::LLRRegularization{TR, N, TI}, x::Union{AbstractArray{T}, AbstractArray{Complex{T}}}, λ::T) where {TR, N, TI, T}
    shape = reg.shape
    blockSize = reg.blockSize
    randshift = reg.randshift
    x = reshape(x, tuple(shape..., length(x) ÷ prod(shape)))

    block_idx = CartesianIndices(blockSize)
    K = size(x)[end]

    if randshift
        # Random.seed!(1234)
        shift_idx = (Tuple(rand(block_idx))..., 0)
        xs = circshift(x, shift_idx)
    else
        xs = x
    end

    ext = mod.(shape, blockSize)
    pad = mod.(blockSize .- ext, blockSize)
    if any(pad .!= 0)
        xp = zeros(eltype(x), (shape .+ pad)..., K)
        xp[CartesianIndices(x)] .= xs
    else
        xp = xs
    end

    bthreads = BLAS.get_num_threads()
    try
        BLAS.set_num_threads(1)
        blocks = CartesianIndices(StepRange.(TI(0), blockSize, shape .- 1))
        xᴸᴸᴿ = [Array{eltype(x)}(undef, prod(blockSize), K) for _ = 1:length(blocks)]
        let xp = xp # Avoid boxing error
            @floop for (id, i) ∈ enumerate(blocks)
                @views xᴸᴸᴿ[id] .= reshape(xp[i.+block_idx, :], :, K)
                ub = sqrt(norm(xᴸᴸᴿ[id]' * xᴸᴸᴿ[id], Inf)) #upper bound on singular values given by matrix infinity norm
                if λ >= ub #save time by skipping the SVT as recommended by Ong/Lustig, IEEE 2016
                    xp[i.+block_idx, :] .= 0
                else # threshold singular values
                    SVDec = svd!(xᴸᴸᴿ[id])
                    prox!(L1Regularization, SVDec.S, λ)
                    xp[i.+block_idx, :] .= reshape(SVDec.U * Diagonal(SVDec.S) * SVDec.Vt, blockSize..., :)
                end
            end
        end
    finally
        BLAS.set_num_threads(bthreads)
    end

    if any(pad .!= 0)
        xs .= xp[CartesianIndices(xs)]
    end

    if randshift
        x .= circshift(xs, -1 .* shift_idx)
    end

    x = vec(x)
    return x
end

"""
    norm(reg::LLRRegularization, x, λ)

returns the value of the LLR-regularization term. The norm is only implemented for 2D, non-fully overlapping blocks. 
"""
function norm(reg::LLRRegularization, x::Union{AbstractArray{T}, AbstractArray{Complex{T}}}, λ::T) where {T <: Real}
    shape = reg.shape
    blockSize = reg.blockSize
    randshift = reg.randshift
    L = reg.L
    Nvoxel = prod(shape)
    K = floor(Int, length(x) / (Nvoxel * L))
    normᴸᴸᴿ = 0.0
    for i = 1:L
        normᴸᴸᴿ += blockNuclearNorm(
            x[(i-1)*Nvoxel*K+1:i*Nvoxel*K],
            shape;
            blockSize = blockSize,
            randshift = randshift,
        )
    end

    return λ * normᴸᴸᴿ
end

function blockNuclearNorm(
    x::Vector{T},
    shape::NTuple{N,TI};
    blockSize::NTuple{N,TI} = ntuple(_ -> 2, N),
    randshift::Bool = true,
    kargs...,
) where {N,T,TI<:Integer}
    x = reshape(x, tuple(shape..., floor(Int64, length(x) / prod(shape))))

    Wy = blockSize[1]
    Wz = blockSize[2]

    if randshift
        srand(1234)
        shift_idx = [rand(1:Wy) rand(1:Wz) 0]
        x = circshift(x, shift_idx)
    end

    ny, nz, K = size(x)

    # reshape into patches
    L = floor(Int, ny * nz / Wy / Wz) # number of patches, assumes that image dimensions are divisble by the blocksizes

    xᴸᴸᴿ = zeros(T, Wy * Wz, L, K)
    for i = 1:K
        xᴸᴸᴿ[:, :, i] = im2colDistinct(x[:, :, i], (Wy, Wz))
    end
    xᴸᴸᴿ = permutedims(xᴸᴸᴿ, [1 3 2])

    # L1-norm of singular values
    normᴸᴸᴿ = 0.0
    for i = 1:L
        SVDec = svd(xᴸᴸᴿ[:, :, i])
        normᴸᴸᴿ += norm(SVDec.S, 1)
    end

    return normᴸᴸᴿ
end


"""
proxLLROverlapping!(reg::LLRRegularization, x, λ)

performs the proximal map for LLR regularization using singular-value-thresholding with fully overlapping blocks
"""
function proxLLROverlapping!(reg::LLRRegularization{TR, N, TI}, x::Union{AbstractArray{T}, AbstractArray{Complex{T}}}, λ::T) where {TR, N, TI, T}
    shape = reg.shape
    blockSize = reg.blockSize
    
    x = reshape(x, tuple(shape..., length(x) ÷ prod(shape)))

    block_idx = CartesianIndices(blockSize)
    K = size(x)[end]

    ext = mod.(shape, blockSize)
    pad = mod.(blockSize .- ext, blockSize)
    if any(pad .!= 0)
        xp = zeros(eltype(x), (shape .+ pad)..., K)
        xp[CartesianIndices(x)] .= x
    else
        xp = copy(x)
    end

    x .= 0 # from here on x is the output

    bthreads = BLAS.get_num_threads()
    try
        BLAS.set_num_threads(1)
        for is ∈ block_idx
            shift_idx = (Tuple(is)..., 0)
            xs = circshift(xp, shift_idx)
            proxLLRNonOverlapping!(reg, xs, λ)
            x .+= circshift(xs, -1 .* shift_idx)[CartesianIndices(x)]
        end
    finally
        BLAS.set_num_threads(bthreads)
    end

    x ./= length(block_idx)
    return vec(x)
end
