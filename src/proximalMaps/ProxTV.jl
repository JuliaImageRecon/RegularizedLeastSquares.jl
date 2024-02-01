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
struct TVRegularization{T, D} <: AbstractParameterizedRegularization{T} where {N, D <: Union{Nothing, Int64, NTuple{N, Int64}}}
  λ::T
  dims::D
  iterations::Int64
end
TVRegularization(λ; dims=nothing, iterations=10, kargs...) = TVRegularization(λ, dims, iterations)

"""
    prox!(reg::TVRegularization, x, λ)

Proximal map for TV regularization. Calculated with the Condat algorithm if the TV is calculated only along one dimension and with the Fast Gradient Projection algorithm otherwise.
"""
prox!(reg::TVRegularization, x::AbstractArray{Tc}, λ::T) where {T,Tc<:Union{T,Complex{T}}} = proxTV!(x, λ, reg.dims, iterations=reg.iterations)
prox!(reg::TVRegularization{T, Nothing}, x::AbstractArray{Tc}, λ::T) where {T,Tc<:Union{T,Complex{T}}} = proxTV!(x, λ, 1:ndims(x), iterations=reg.iterations)

function proxTV!(x::AbstractArray{T}, λ::T, dims::Integer; kwargs...) where {T<:Real}
  shape = size(x)
  i = CartesianIndices((ones(Int, dims - 1)..., 0:shape[dims]-1, ones(Int, length(shape) - dims)...))

  Threads.@threads for j ∈ CartesianIndices((shape[1:dims-1]..., 1, shape[dims+1:end]...))
    @views @inbounds tv_denoise_1d_condat!(x[j.+i], shape[dims], λ)
  end
  return x
end

function proxTV!(x::AbstractArray{Tc}, λ::T, dims = 1:ndims(x); iterations=10) where {T<:Real,Tc<:Union{T,Complex{T}}}
  shape = size(x)
  ∇ = GradientOp(Tc; shape, dims)

  # allocate these in reg term and reuse them?
  # allocate storage
  xTmp = similar(vec(x))
  pq = similar(xTmp, size(∇, 1))
  rs = similar(pq)
  pqOld = similar(pq)

  @assert length(xTmp) == length(x)
  # initialize dual variables
  xTmp .= 0
  pq .= 0
  rs .= 0
  pqOld .= 0

  t = one(T)
  for _ = 1:iterations
    pqTmp = pqOld
    pqOld = pq
    pq = rs

    # gradient projection step for dual variables
    Threads.@threads for i ∈ eachindex(xTmp, x)
      @inbounds xTmp[i] = x[i]
    end
    mul!(xTmp, transpose(∇), rs, -λ, 1) # xtmp = x-λ*transpose(∇)*rs
    mul!(pq, ∇, xTmp, 1 / (8λ), 1) # rs = ∇*xTmp/(8λ)

    restrictMagnitude!(pq)

    # form linear combination of old and new estimates
    tOld = t
    t = (1 + sqrt(1 + 4 * tOld^2)) / 2
    t2 = ((tOld - 1) / t)
    t3 = 1 + t2

    rs = pqTmp
    Threads.@threads for i ∈ eachindex(rs, pq, pqOld)
      @inbounds rs[i] = t3 * pq[i] - t2 * pqOld[i]
    end
  end

  mul!(vec(x), transpose(∇), pq, -λ, one(Tc)) # x .-= λ*transpose(∇)*pq
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
function norm(reg::TVRegularization, x::AbstractArray{Tc}, λ::T) where {T<:Real,Tc<:Union{T,Complex{T}}}
  ∇ = GradientOp(Tc; shape=size(x), dims= isnothing(reg.dims) ? UnitRange(1, ndims(x)) : reg.dims)
  return λ * norm(∇ * vec(x), 1)
end
