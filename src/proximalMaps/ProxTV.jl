export proxTV!, normTV

mutable struct TVParams{Tc, matT}
  pq::Vector{Tc}
  rs::Vector{Tc}
  pqOld::Vector{Tc}
  xTmp::Vector{Tc}
  ∇::matT
end

function TVParams(shape::NTuple{N,Int}, T::Type=Float64; dim::Union{Int,Nothing}=nothing, kargs...) where N 
  return TVParams( Array{T}(undef,prod(shape)); shape=shape, dim=dim)
end

function TVParams(x::Vector{Tc}; shape::NTuple{N,Int}=(), dim::Union{Int,Nothing}=nothing, kargs...) where {Tc,N}
  # gradient operator
  if dim !== nothing
    ∇ = GradientOp(Tc,shape, dim)
  else
    ∇ = GradientOp(Tc,shape)
  end
  # allocate storage
  xTmp = similar(x)
  pq = similar(x, size(∇,1))
  rs = similar(pq)
  pqOld = similar(pq)
  
  return TVParams(pq, rs, pqOld, xTmp, ∇)
end



"""
    proxTV!(x::Vector{T}, λ::Float64; kargs...) where T

proximal map for ansitropic TV regularization using the Fast Gradient Projection algorithm.

Reference for the FGP algorithm:
A. Beck and T. Teboulle, "Fast Gradient-Based Algorithms for Constrained
Total Variation Image Denoising
and Deblurring Problems", IEEE Trans. Image Process. 18(11), 2009

# Arguments
* `x::Array{T}`             - Vector to apply proximal map to
* `λ::Float64`              - regularization paramter
* `shape::NTuple=[]`        - size of the underlying image
* `iterationsTV::Int64=20`  - number of FGP iterations
"""
function proxTV!(x::Vector{Tc}, λ::T; shape::NTuple{N,Int}=(), dim::Union{Int,Nothing}=nothing, iterationsTV::Int64=20, tvpar::Union{TVParams{Tc},Nothing}=nothing, kargs...) where {T,Tc,N}
  if tvpar===nothing
    tvpar = TVParams(x; shape=shape, dim=dim)
  end

  # apply TV prox map
  proxTV!(x,λ,tvpar; iterationsTV=iterationsTV, kargs...)
end

function proxTV!(x::Vector{Tc}, λ::T, p::TVParams{Tc}; iterationsTV::Int64=20, kargs...) where {T,Tc,N}

  # initialize dual variables
  p.xTmp .= zero(Tc)
  p.pq .= zero(Tc)
  p.rs .= zero(Tc)
  p.pqOld .= zero(Tc)

  t = 1
  for i=1:iterationsTV
    p.pqOld .= p.pq

    # gradient projection step for dual variables
    p.xTmp .= x
    mul!(p.xTmp, transpose(p.∇), p.rs, -λ, one(Tc)) # xtmp = x-λ*transpose(∇)*rs
    p.pq .= p.rs
    mul!(p.pq, p.∇, p.xTmp, one(Tc)/(8λ), one(Tc)) # rs = ∇*xTmp/(8λ)
    restrictMagnitude!(p.pq)

    # form linear combinaion of old and new estimates
    tOld = t
    t = ( 1+sqrt(1+4*tOld^2) )/2
    p.rs .= p.pq .+ (tOld-1)/t*(p.pq.-p.pqOld)
  end

  mul!(x, transpose(p.∇), p.pq, -λ, one(Tc)) # x .-= λ*transpose(∇)*pq
end

# restrict x to a number smaller then one
function restrictMagnitude!(x::Array)
  x ./= max.(1.0, abs.(x))
end

"""
    normTV(x::Vector{T},λ::Float64;shape::NTuple=[],dim=nothing,kargs...) where T

returns the value of the ansisotropic TV-regularization term.
Arguments are the same as in `proxTV!`
"""
# function normTV(x::Vector{Tc},λ::T;shape::NTuple=[],kargs...) where {T,Tc}
#   x = reshape(x,shape)
#   tv = norm(vec(x[1:end-1,1:end-1]-x[2:end,1:end-1]),1)
#   tv += norm(vec(x[1:end-1,1:end-1]-x[1:end-1,2:end]),1)
#   tv += norm(vec(x[1:end-1,end]-x[2:end,end]),1)
#   tv += norm(vec(x[end,1:end-1]-x[end,2:end]),1)
#   return λ*tv
# end
function normTV(x::Vector{Tc},λ::T;shape::NTuple=[], dim::Union{Int,Nothing}=nothing, kargs...) where {T,Tc}
  if dim !== nothing
    ∇ = GradientOp(Tc,shape, dim)
  else
    ∇ = GradientOp(Tc,shape)
  end

  return λ*norm(∇*x,1)
end
