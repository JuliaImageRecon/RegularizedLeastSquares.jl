export L2Regularization, proxL2!, normL2

struct L2Regularization{T} <: AbstractRegularization{T}
  λ::T
end

"""
    proxL2!(x::Vector{T}, λ::Float64; kargs...) where T

proximal map for Tikhonov regularization.
"""
proxL2!(x, λ; kargs...) = prox!(L2Regularization, x, λ; kargs...)
function prox!(::Type{<:L2Regularization}, x::T, λ::Float64; kargs...) where T<:AbstractArray
  x[:] .*= 1. / (1. + 2. *λ)#*x
end

"""
    normL2(x::Vector{T}, λ::Float64, kargs...)

returns the value of the L2-regularization term
"""
normL2(x, λ; kargs...) = norm(L2Regularization, x, λ; kargs...)
norm(::Type{<:L2Regularization}, x::T, λ::Float64; kargs...) where T<:AbstractArray = λ*norm(x,2)^2
