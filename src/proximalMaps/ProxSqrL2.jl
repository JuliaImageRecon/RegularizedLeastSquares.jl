export SqrL2Regularization

"""
    SqrL2Regularization

Regularization term implementing the proximal map for Tikhonov regularization.
"""
struct SqrL2Regularization{T} <: AbstractParameterizedRegularization{T}
  λ::T
  SqrL2Regularization(λ::T; kargs...) where T = new{T}(λ)
end

"""
    prox!(reg::SqrL2Regularization, x, λ)

performs the proximal map for Tikhonov regularization.
"""
function prox!(::SqrL2Regularization, x::AbstractArray{Tc}, λ::T) where {T, Tc <: Union{T, Complex{T}}}
  x[:] .*= 1. / (1. + 2. *λ)#*x
  return x
end

"""
    norm(reg::SqrL2Regularization, x, λ)

returns the value of the L2-regularization term
"""
norm(::SqrL2Regularization, x::AbstractArray{Tc}, λ::T) where {T, Tc <: Union{T, Complex{T}}} = λ*norm(x,2)^2
