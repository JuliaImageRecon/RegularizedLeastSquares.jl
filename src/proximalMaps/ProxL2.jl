export L2Regularization

"""
    L2Regularization

Regularization term implementing the proximal map for Tikhonov regularization.
"""
struct L2Regularization{T} <: AbstractParameterizedRegularization{T}
  λ::T
  L2Regularization(λ::T; kargs...) where T = new{T}(λ)
end

"""
    prox!(reg::L2Regularization, x, λ)

performs the proximal map for Tikhonov regularization.
"""
function prox!(::L2Regularization, x::Union{AbstractArray{T}, AbstractArray{Complex{T}}}, λ::T) where {T}
  x[:] .*= 1. ./ (1. .+ 2. .*λ)#*x
  return x
end

"""
    norm(reg::L2Regularization, x, λ)

returns the value of the L2-regularization term
"""
norm(::L2Regularization, x::Union{AbstractArray{T}, AbstractArray{Complex{T}}}, λ::T) where {T} = λ*norm(x,2)^2
function norm(::L2Regularization, x::Union{AbstractArray{T}, AbstractArray{Complex{T}}}, λ::AbstractArray{T}) where {T}
  res = zero(real(eltype(x)))
  for i in eachindex(x)
    res+= λ[i]*abs2(x[i])
  end
  return res
end