export proxTV!, normTV

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
  proxTV!(x::Vector{Tc}, λ::T; shape::NTuple{N,Int}, dims=1:length(shape), iterationsTV=20, tvpar=TVParams(x; shape=shape, dims=dims))

proximal map for ansitropic TV regularization using the Fast Gradient Projection algorithm.

Reference for the FGP algorithm:
A. Beck and T. Teboulle, "Fast Gradient-Based Algorithms for Constrained
Total Variation Image Denoising
and Deblurring Problems", IEEE Trans. Image Process. 18(11), 2009

# Arguments
* `x::Array{Tc}`            - Vector to apply proximal map to
* `λ::T`                    - regularization paramter
* `shape::NTuple`           - size of the underlying image
* `iterationsTV::Int64=20`  - number of FGP iterations
"""
function proxTV!(x::AbstractVector{Tc}, λ::T; shape, dims=1:length(shape), iterationsTV=20, tvpar=TVParams(x; shape=shape, dims=dims)) where {T <: Real,Tc <: Union{T, Complex{T}}}
  proxTV!(x,λ,tvpar; iterationsTV=iterationsTV)
end

function proxTV!(x::AbstractVector{Tc}, λ::T, p::TVParams{Tc}; iterationsTV=20) where {T <: Real, Tc <: Union{T, Complex{T}}}
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

    # form linear combinaion of old and new estimates
    tOld = t
    t = (1 + sqrt(1+4*tOld^2)) / 2
    t2 = ((tOld-1)/t)
    p.rs .= (1+t2) .* p.pq .- t2 .* p.pqOld
  end

  mul!(x, transpose(p.∇), p.pq, -λ, one(Tc)) # x .-= λ*transpose(∇)*pq
end

# restrict x to a number smaller then one
function restrictMagnitude!(x)
  x ./= max.(1, abs.(x))
end

"""
  normTV(x::Vector{Tc},λ::T; shape, dims=1:length(shape))

returns the value of the ansisotropic TV-regularization term.
Arguments are the same as in `proxTV!`
"""
function normTV(x::Vector{Tc},λ::T; shape, dims=1:length(shape)) where {T <: Real, Tc <: Union{T, Complex{T}}}
  ∇ = GradientOp(Tc,shape, dims)
  return λ * norm(∇*x, 1)
end