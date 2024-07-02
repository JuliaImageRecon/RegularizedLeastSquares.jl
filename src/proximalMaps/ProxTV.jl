export TVRegularization

mutable struct TVParams{Tc,vecTc <: AbstractVector{Tc}, matT}
  pq::vecTc
  rs::vecTc
  pqOld::vecTc
  xTmp::vecTc
  ∇::matT
end

"""
    TVRegularization

Regularization term implementing the proximal map for TV regularization. Calculated with the Condat algorithm if the TV is calculated only along one real-valued dimension and with the Fast Gradient Projection algorithm otherwise.

Reference for the Condat algorithm:
https://lcondat.github.io/publis/Condat-fast_TV-SPL-2013.pdf

Reference for the FGP algorithm:
A. Beck and T. Teboulle, "Fast Gradient-Based Algorithms for Constrained
Total Variation Image Denoising
and Deblurring Problems", IEEE Trans. Image Process. 18(11), 2009

# Arguments
* `λ::T`                    - regularization parameter

# Keywords
* `shape::NTuple`           - size of the underlying image
* `dims`                    - Dimension to perform the TV along. If `Integer`, the Condat algorithm is called, and the FDG algorithm otherwise.
* `iterationsTV=20`         - number of FGP iterations
"""
mutable struct TVRegularization{T,N,TI} <: AbstractParameterizedRegularization{T} where {N,TI<:Integer}
  λ::T
  dims
  shape::NTuple{N,TI}
  iterationsTV::Int64
  params::Union{TVParams, Nothing}
end
TVRegularization(λ; shape=(0,), dims=1:length(shape), iterationsTV=10, kargs...) = TVRegularization(λ, dims, shape, iterationsTV, nothing)

function TVParams(shape, T::Type=Float64; dims=1:length(shape))
  return TVParams(Vector{T}(undef, prod(shape)); shape=shape, dims=dims)
end

function TVParams(x::AbstractVector{Tc}; shape, dims=1:length(shape)) where {Tc}
  ∇ = GradientOp(Tc; shape, dims, S = typeof(x))

  # allocate storage
  xTmp = similar(x)
  pq = similar(x, size(∇, 1))
  rs = similar(pq)
  pqOld = similar(pq)

  return TVParams(pq, rs, pqOld, xTmp, ∇)
end



"""
    prox!(reg::TVRegularization, x, λ)

Proximal map for TV regularization. Calculated with the Condat algorithm if the TV is calculated only along one dimension and with the Fast Gradient Projection algorithm otherwise.
"""
prox!(reg::TVRegularization, x::Union{AbstractArray{T}, AbstractArray{Complex{T}}}, λ::T) where {T <: Real} = proxTV!(reg, x, λ, shape=reg.shape, dims=reg.dims, iterationsTV=reg.iterationsTV)

function proxTV!(reg, x, λ; shape, dims=1:length(shape), kwargs...) # use kwargs for shape and dims
  return proxTV!(reg, x, λ, shape, dims; kwargs...) # define shape and dims w/o kwargs to enable multiple dispatch on dims
end

proxTV!(reg, x, shape, dims::Integer; kwargs...) = proxTV!(reg, x, shape, dims; kwargs...)
function proxTV!(x::AbstractVector{T}, λ::T, shape, dims::Integer; kwargs...) where {T<:Real}
  x_ = reshape(x, shape)
  i = CartesianIndices((ones(Int, dims - 1)..., 0:shape[dims]-1, ones(Int, length(shape) - dims)...))

  Threads.@threads for j ∈ CartesianIndices((shape[1:dims-1]..., 1, shape[dims+1:end]...))
    @views @inbounds tv_denoise_1d_condat!(x_[j.+i], shape[dims], λ)
  end
  return x
end

# Reuse TvParams if possible
function proxTV!(reg, x::AbstractVector{Tc}, λ::T, shape, dims; iterationsTV=10, kwargs...) where {T<:Real,Tc<:Union{T,Complex{T}}}
  if isnothing(reg.params) || length(x) != length(reg.params.xTmp) || typeof(x) != typeof(reg.params.xTmp)
    reg.params = TVParams(x; shape = shape, dims = dims)
  end
  return proxTV!(x, λ, reg.params; iterationsTV=iterationsTV)
end

function proxTV!(x::AbstractVector{Tc}, λ::T, p::TVParams{Tc}; iterationsTV=10, kwargs...) where {T<:Real,Tc<:Union{T,Complex{T}}}
  @assert length(p.xTmp) == length(x)
  @assert length(p.rs) == length(p.pq)
  @assert length(p.rs) == length(p.pq)

  # initialize dual variables
  p.xTmp .= 0
  p.pq .= 0
  p.rs .= 0
  p.pqOld .= 0

  t = one(T)
  for _ = 1:iterationsTV
    pqTmp = p.pqOld
    p.pqOld = p.pq
    p.pq = p.rs

    # gradient projection step for dual variables
    tv_copy!(p.xTmp, x)
    mul!(p.xTmp, transpose(p.∇), p.rs, -λ, 1) # xtmp = x-λ*transpose(∇)*rs
    mul!(p.pq, p.∇, p.xTmp, 1 / (8λ), 1) # rs = ∇*xTmp/(8λ)

    tv_restrictMagnitude!(p.pq)

    # form linear combination of old and new estimates
    tOld = t
    t = (1 + sqrt(1 + 4 * tOld^2)) / 2
    t2 = ((tOld - 1) / t)
    t3 = 1 + t2

    p.rs = pqTmp
    tv_linearcomb!(p.rs, t3, p.pq, t2, p.pqOld)
  end

  mul!(x, transpose(p.∇), p.pq, -λ, one(Tc)) # x .-= λ*transpose(∇)*pq
  return x
end

tv_copy!(dest, src) = copyto!(dest, src)
function tv_copy!(dest::Vector{T}, src::Vector{T}) where T
  Threads.@threads for i ∈ eachindex(dest, src)
    @inbounds dest[i] = src[i]
  end
end

# restrict x to a number smaller then one
function tv_restrictMagnitude!(x)
  Threads.@threads for i in eachindex(x)
    @inbounds x[i] /= max(1, abs(x[i]))
  end
end

function tv_linearcomb!(rs, t3, pq, t2, pqOld)
  Threads.@threads for i ∈ eachindex(rs, pq, pqOld)
    @inbounds rs[i] = t3 * pq[i] - t2 * pqOld[i]
  end
end

"""
  norm(reg::TVRegularization, x, λ)

returns the value of the TV-regularization term.
"""
function norm(reg::TVRegularization, x::Union{AbstractArray{T}, AbstractArray{Complex{T}}}, λ::T) where {T <: Real}
  ∇ = GradientOp(eltype(x); shape=reg.shape, dims=reg.dims)
  return λ * norm(∇ * x, 1)
end
