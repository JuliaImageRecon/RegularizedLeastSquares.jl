export L2Regularization, proxL2!, normL2

struct L2Regularization{T} <: AbstractParameterizedRegularization{T}
  λ::T
  L2Regularization(λ::T; kargs...) where T = new{T}(λ)
end

"""
    proxL2!(x::Vector{T}, λ::Float64; kargs...) where T

proximal map for Tikhonov regularization.
"""
proxL2!(x, λ; kargs...) = prox!(L2Regularization, x, λ; kargs...)
function prox!(::Type{<:L2Regularization}, x::AbstractArray{Tc}, λ::T; kargs...) where {T, Tc <: Union{T, Complex{T}}}
  x[:] .*= 1. / (1. + 2. *λ)#*x
end

"""
    normL2(x::Vector{T}, λ::Float64, kargs...)

returns the value of the L2-regularization term
"""
normL2(x, λ; kargs...) = norm(L2Regularization, x, λ; kargs...)
norm(::Type{<:L2Regularization}, x::AbstractArray{Tc}, λ::T; kargs...) where {T, Tc <: Union{T, Complex{T}}} = λ*norm(x,2)^2
