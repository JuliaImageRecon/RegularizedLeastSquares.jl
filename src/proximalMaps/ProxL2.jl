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
function prox!(::L2Regularization, x::AbstractArray{Tc}, λ::T) where {T, Tc <: Union{T, Complex{T}}}
  x[:] .*= 1. / (1. + 2. *λ)#*x
end

"""
    norm(reg::L2Regularization, x, λ)

returns the value of the L2-regularization term
"""
norm(::L2Regularization, x::AbstractArray{Tc}, λ::T) where {T, Tc <: Union{T, Complex{T}}} = λ*norm(x,2)^2
