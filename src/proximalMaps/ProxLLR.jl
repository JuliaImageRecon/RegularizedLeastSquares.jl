export LLRRegularization

"""
    LLRRegularization

Regularization term implementing the proximal map for locally low rank (LLR) regularization using singular-value-thresholding.

# Arguments
* `λ`                  - regularization paramter
* `shape::Tuple{Int}=[]`        - dimensions of the image
* `blockSize::Tuple{Int}=[2;2]` - size of patches to perform singular value thresholding on
* `randshift::Bool=true`        - randomly shifts the patches to ensure translation invariance
"""
struct LLRRegularization{T, N, TI} <: AbstractParameterizedRegularization{T} where {N, TI<:Integer}
  λ::T
  shape::NTuple{N,TI}
  blockSize::NTuple{N,TI}
  randshift::Bool
  L::Int64
end
LLRRegularization(λ;  shape::NTuple{N,TI}, blockSize::NTuple{N,TI} = ntuple(_ -> 2, N), randshift::Bool = true, L::Int64 = 1, kargs...) where {N,TI<:Integer} =
 LLRRegularization(λ, shape, blockSize, randshift, L)

"""
    prox!(reg::LLRRegularization, x, λ)

performs the proximal map for LLR regularization using singular-value-thresholding
"""
function prox!(reg::LLRRegularization{TR, N, TI}, x::AbstractArray{Tc}, λ::T) where {TR, N, TI, T, Tc <: Union{T, Complex{T}}}
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
        xp = zeros(Tc, (shape .+ pad)..., K)
        xp[CartesianIndices(x)] .= xs
    else
        xp = xs
    end

    bthreads = BLAS.get_num_threads()
    try
        BLAS.set_num_threads(1)
        xᴸᴸᴿ = [Array{Tc}(undef, prod(blockSize), K) for _ = 1:Threads.nthreads()]
        let xp = xp # Avoid boxing error
            @floop for i ∈ CartesianIndices(StepRange.(TI(0), blockSize, shape .- 1))
                @views xᴸᴸᴿ[Threads.threadid()] .= reshape(xp[i.+block_idx, :], :, K)
                ub = sqrt(norm(xᴸᴸᴿ[Threads.threadid()]' * xᴸᴸᴿ[Threads.threadid()], Inf)) #upper bound on singular values given by matrix infinity norm
                if λ >= ub #save time by skipping the SVT as recommended by Ong/Lustig, IEEE 2016
                    xp[i.+block_idx, :] .= 0
                else # threshold singular values
                    SVDec = svd!(xᴸᴸᴿ[Threads.threadid()])
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

returns the value of the LLR-regularization term.
"""
function norm(reg::LLRRegularization, x::Vector{Tc}, λ::T) where {T, Tc <: Union{T, Complex{T}}}
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
proxLLROverlapping!(x::Vector{T}, λ=1e-6; kargs...) where T

proximal map for LLR regularization with fully overlapping blocks

# Arguments
* `x::Vector{T}`                - Vector to apply proximal map to
* `λ`                           - regularization parameter
* `shape::Tuple{Int}=[]`        - dimensions of the image
* `blockSize::NTuple{Int}=ntuple(_ -> 2, N)` - size of patches to perform singular value thresholding on
"""
function proxLLROverlapping!(
        x::Vector{T},
        λ;
        shape::NTuple{N,TI},
        blockSize::NTuple{N,TI} = ntuple(_ -> 2, N),
    ) where {T,N,TI<:Integer}

    x = reshape(x, tuple(shape..., length(x) ÷ prod(shape)))

    block_idx = CartesianIndices(blockSize)
    K = size(x)[end]

    ext = mod.(shape, blockSize)
    pad = mod.(blockSize .- ext, blockSize)
    if any(pad .!= 0)
        xp = zeros(T, (shape .+ pad)..., K)
        xp[CartesianIndices(x)] .= x
    else
        xp = copy(x)
    end

    x .= 0 # from here on x is the output

    bthreads = BLAS.get_num_threads()
    try
        BLAS.set_num_threads(1)
        xᴸᴸᴿ = [Array{T}(undef, prod(blockSize), K) for _ = 1:Threads.nthreads()]
        for is ∈ block_idx
            shift_idx = (Tuple(is)..., 0)
            xs = circshift(xp, shift_idx)

            @floop for i ∈ CartesianIndices(StepRange.(TI(0), blockSize, shape .- 1))
                @views xᴸᴸᴿ[Threads.threadid()] .= reshape(xs[i.+block_idx, :], :, K)

                ub = sqrt(norm(xᴸᴸᴿ[Threads.threadid()]' * xᴸᴸᴿ[Threads.threadid()], Inf)) #upper bound on singular values given by matrix infinity norm
                if λ >= ub #save time by skipping the SVT as recommended by Ong/Lustig, IEEE 2016
                    xs[i.+block_idx, :] .= 0
                else # threshold singular values
                    SVDec = svd!(xᴸᴸᴿ[Threads.threadid()])
                    prox!(L1Regularization, SVDec.S, λ)
                    xs[i.+block_idx, :] .= reshape(SVDec.U * Diagonal(SVDec.S) * SVDec.Vt, blockSize..., :)
                end
            end
            x .+= circshift(xs, -1 .* shift_idx)[CartesianIndices(x)]
        end
    finally
        BLAS.set_num_threads(bthreads)
    end

    x ./= length(block_idx)
    return vec(x)
end
