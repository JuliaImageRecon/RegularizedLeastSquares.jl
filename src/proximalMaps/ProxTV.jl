export TVRegularization, proxTV!, normTV

struct TVRegularization{T} <: AbstractRegularization{T}
  λ::T
  dims
  shape::Union{Nothing, Vector{Int64}}
  iterationsTV::Int64
end
TVRegularization(λ; shape = nothing, dims = isnothing(shape) ? 0 : 1:length(shape), iterationsTV = 10, kargs...) = TVRegularization(λ, dims, shape, iterationsTV)


mutable struct TVParams{Tc, matT}
  pq::Vector{Tc}
  rs::Vector{Tc}
  pqOld::Vector{Tc}
  xTmp::Vector{Tc}
  ∇::matT
end

function TVParams(shape, T::Type=Float64; dims=1:length(shape))
  return TVParams(Vector{T}(undef,prod(shape)); shape=shape, dims=dims)
end

function TVParams(x::AbstractVector{Tc}; shape, dims=1:length(shape)) where {Tc}
  ∇ = GradientOp(Tc,shape, dims)

  # allocate storage
  xTmp = similar(x)
  pq = similar(x, size(∇,1))
  rs = similar(pq)
  pqOld = similar(pq)

  return TVParams(pq, rs, pqOld, xTmp, ∇)
end



"""
  proxTV!(x::Vector{Tc}, λ::T; shape::NTuple{N,Int}, dims, iterationsTV=20, tvpar=TVParams(x; shape=shape, dims=dims))

Proximal map for TV regularization. Calculated with the Condat algorithm if the TV is calculated only along one dimension and with the Fast Gradient Projection algorithm otherwise.

Reference for the Condat algorithm:
https://lcondat.github.io/publis/Condat-fast_TV-SPL-2013.pdf

Reference for the FGP algorithm:
A. Beck and T. Teboulle, "Fast Gradient-Based Algorithms for Constrained
Total Variation Image Denoising
and Deblurring Problems", IEEE Trans. Image Process. 18(11), 2009

Reference for the FGP algorithm:
A. Beck and T. Teboulle, "Fast Gradient-Based Algorithms for Constrained
Total Variation Image Denoising
and Deblurring Problems", IEEE Trans. Image Process. 18(11), 2009

# Arguments
* `x::Array{Tc}`            - Vector to apply proximal map to
* `λ::T`                    - regularization parameter
* `shape::NTuple`           - size of the underlying image
* `dims`                    - Dimension to perform the TV along. If `Integer`, the Condat algorithm is called, and the FDG algorithm otherwise.
* `iterationsTV=20`         - number of FGP iterations
"""
proxTV!(x, λ; kargs...) = prox!(TVRegularization, x, λ; kargs...)
function prox!(::Type{<:TVRegularization}, x, λ; shape, dims=1:length(shape), kwargs...) # use kwargs for shape and dims
  return proxTV!(x, λ, shape, dims; kwargs...) # define shape and dims w/o kwargs to enable multiple dispatch on dims
end

function proxTV!(x::AbstractVector{T}, λ::T, shape, dims::Integer; kwargs...) where {T <: Real}
  x_ = reshape(x, shape)

  if dims == 1
    for j ∈ CartesianIndices(shape[dims+1:end])
      @views @inbounds tv_denoise_1d_condat!(x_[:,j], shape[dims], λ)
    end
  elseif dims == length(shape)
    for i ∈ CartesianIndices(shape[1:dims-1])
      @views @inbounds tv_denoise_1d_condat!(x_[i,:], shape[dims], λ)
    end
  else
    for j ∈ CartesianIndices(shape[dims+1:end]), i ∈ CartesianIndices(shape[1:dims-1])
      @views @inbounds tv_denoise_1d_condat!(x_[i,:,j], shape[dims], λ)
    end
  end
  return x
end

# reinterpret complex-valued vector as 2xN matrix and change the shape etc accordingly
function proxTV!(x::AbstractVector{Tc}, λ::T, shape, dims::Integer; kwargs...) where {T <: Real, Tc <: Complex{T}}
  proxTV!(vec(reinterpret(reshape, T, x)), λ, shape=(2, shape...), dims=(dims+1))
  return x
end

function proxTV!(x::AbstractVector{Tc}, λ::T, shape, dims; iterationsTV=20, tvpar=TVParams(x; shape=shape, dims=dims), kwargs...) where {T <: Real,Tc <: Union{T, Complex{T}}}
  return proxTV!(x,λ,tvpar; iterationsTV=iterationsTV)
end

function proxTV!(x::AbstractVector{Tc}, λ::T, p::TVParams{Tc}; iterationsTV=20, kwargs...) where {T <: Real, Tc <: Union{T, Complex{T}}}
  # initialize dual variables
  p.xTmp  .= 0
  p.pq    .= 0
  p.rs    .= 0
  p.pqOld .= 0

  t = one(T)
  for _ = 1:iterationsTV
    p.pqOld .= p.pq

    # gradient projection step for dual variables
    p.xTmp .= x
    mul!(p.xTmp, transpose(p.∇), p.rs, -λ, 1) # xtmp = x-λ*transpose(∇)*rs
    p.pq .= p.rs
    mul!(p.pq, p.∇, p.xTmp, 1/(8λ), 1) # rs = ∇*xTmp/(8λ)
    restrictMagnitude!(p.pq)

    # form linear combination of old and new estimates
    tOld = t
    t = (1 + sqrt(1+4*tOld^2)) / 2
    t2 = ((tOld-1)/t)
    p.rs .= (1+t2) .* p.pq .- t2 .* p.pqOld
  end

  mul!(x, transpose(p.∇), p.pq, -λ, one(Tc)) # x .-= λ*transpose(∇)*pq
  return x
end

# restrict x to a number smaller then one
function restrictMagnitude!(x)
  x ./= max.(1, abs.(x))
end

"""
  normTV(x::Vector{Tc},λ::T; shape, dims=1:length(shape))

returns the value of the TV-regularization term.
Arguments are the same as in `proxTV!`
"""
normTV(x, λ; kargs...) = norm(TVRegularization, x, λ; kargs...)
function norm(::Type{<:TVRegularization}, x::Vector{Tc},λ::T; shape, dims=1:length(shape)) where {T <: Real, Tc <: Union{T, Complex{T}}}
  ∇ = GradientOp(Tc,shape, dims)
  return λ * norm(∇*x, 1)
end
