export L2Regularization

"""
    L2Regularization

Regularization term implementing the proximal map for Tikhonov regularization.
"""
struct L2Regularization{T} <: AbstractParameterizedRegularization{T}
  λ::T
  L2Regularization(λ::T; kargs...) where T = new{T}(λ)
end
Adapt.adapt_structure(to, reg::L2Regularization) = L2Regularization(Adapt.adapt(to, reg.λ))

"""
    prox!(reg::L2Regularization, x, λ)

performs the proximal map for Tikhonov regularization.
"""
function prox!(::L2Regularization, x::Union{AbstractArray{T}, AbstractArray{Complex{T}}}, λ::T) where {T <: Real}
  x[:] .*= 1. ./ (1. .+ 2. .*λ)#*x
  return x
end

"""
    norm(reg::L2Regularization, x, λ)

returns the value of the L2-regularization term
"""
norm(::L2Regularization, x::Union{AbstractArray{T}, AbstractArray{Complex{T}}}, λ::T) where {T <: Real} = λ*norm(x,2)^2
function norm(::L2Regularization, x::Union{AbstractArray{T}, AbstractArray{Complex{T}}}, λ::AbstractArray{T}) where {T <: Real}
  return mapreduce((w, xi) -> w * abs2(xi), +, view(vec(λ), 1:length(x)), vec(x); init = zero(real(eltype(x))))
end