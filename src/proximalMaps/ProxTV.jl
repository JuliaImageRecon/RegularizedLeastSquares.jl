export TVRegularization

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
struct TVRegularization{T,N,TI} <: AbstractParameterizedRegularization{T} where {N,TI<:Integer}
  λ::T
  dims
  shape::NTuple{N,TI}
  iterationsTV::Int64
end
TVRegularization(λ; shape=(0,), dims=1:length(shape), iterationsTV=10, kargs...) = TVRegularization(λ, dims, shape, iterationsTV)


mutable struct TVParams{Tc,matT}
  pq::Vector{Tc}
  rs::Vector{Tc}
  pqOld::Vector{Tc}
  xTmp::Vector{Tc}
  ∇::matT
end

function TVParams(shape, T::Type=Float64; dims=1:length(shape))
  return TVParams(Vector{T}(undef, prod(shape)); shape=shape, dims=dims)
end

function TVParams(x::AbstractVector{Tc}; shape, dims=1:length(shape)) where {Tc}
  ∇ = GradientOp(Tc; shape, dims)

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
prox!(reg::TVRegularization, x::Vector{Tc}, λ::T) where {T,Tc<:Union{T,Complex{T}}} = proxTV!(x, λ, shape=reg.shape, dims=reg.dims, iterationsTV=reg.iterationsTV)

function proxTV!(x, λ; shape, dims=1:length(shape), kwargs...) # use kwargs for shape and dims
  return proxTV!(x, λ, shape, dims; kwargs...) # define shape and dims w/o kwargs to enable multiple dispatch on dims
end

function proxTV!(x::AbstractVector{T}, λ::T, shape, dims::Integer; kwargs...) where {T<:Real}
  x_ = reshape(x, shape)
  i = CartesianIndices((ones(Int, dims - 1)..., 0:shape[dims]-1, ones(Int, length(shape) - dims)...))

  Threads.@threads for j ∈ CartesianIndices((shape[1:dims-1]..., 1, shape[dims+1:end]...))
    @views @inbounds tv_denoise_1d_condat!(x_[j.+i], shape[dims], λ)
  end
  return x
end

function proxTV!(x::AbstractVector{Tc}, λ::T, shape, dims; iterationsTV=10, tvpar=TVParams(x; shape=shape, dims=dims), kwargs...) where {T<:Real,Tc<:Union{T,Complex{T}}}
  return proxTV!(x, λ, tvpar; iterationsTV=iterationsTV)
end

function proxTV!(x::AbstractVector{Tc}, λ::T, p::TVParams{Tc}; iterationsTV=10, kwargs...) where {T<:Real,Tc<:Union{T,Complex{T}}}
  @assert length(p.xTmp) == length(x)
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
    Threads.@threads for i ∈ eachindex(p.xTmp, x)
      @inbounds p.xTmp[i] = x[i]
    end
    mul!(p.xTmp, transpose(p.∇), p.rs, -λ, 1) # xtmp = x-λ*transpose(∇)*rs
    mul!(p.pq, p.∇, p.xTmp, 1 / (8λ), 1) # rs = ∇*xTmp/(8λ)

    restrictMagnitude!(p.pq)

    # form linear combination of old and new estimates
    tOld = t
    t = (1 + sqrt(1 + 4 * tOld^2)) / 2
    t2 = ((tOld - 1) / t)
    t3 = 1 + t2

    p.rs = pqTmp
    Threads.@threads for i ∈ eachindex(p.rs, p.pq, p.pqOld)
      @inbounds p.rs[i] = t3 * p.pq[i] - t2 * p.pqOld[i]
    end
  end

  mul!(x, transpose(p.∇), p.pq, -λ, one(Tc)) # x .-= λ*transpose(∇)*pq
  return x
end

# restrict x to a number smaller then one
function restrictMagnitude!(x)
  Threads.@threads for i in eachindex(x)
    @inbounds x[i] /= max(1, abs(x[i]))
  end
end

"""
  norm(reg::TVRegularization, x, λ)

returns the value of the TV-regularization term.
"""
function norm(reg::TVRegularization, x::Vector{Tc}, λ::T) where {T<:Real,Tc<:Union{T,Complex{T}}}
  ∇ = GradientOp(Tc; shape=reg.shape, dims=reg.dims)
  return λ * norm(∇ * x, 1)
end
