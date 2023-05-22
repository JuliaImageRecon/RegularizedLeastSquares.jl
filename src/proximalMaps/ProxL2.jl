export L2Regularization, proxL2!, normL2

struct L2Regularization <: AbstractRegularization
  λ::Float64
end

prox!(reg::L2Regularization, x) = proxL2!(x, reg.λ)
norm(reg::L2Regularization, x) = normL2(x, reg.λ)


"""
    proxL2!(x::Vector{T}, λ::Float64; kargs...) where T

proximal map for Tikhonov regularization.
"""
function proxL2!(x::T, λ::Float64; kargs...) where T<:AbstractArray
  x[:] .*= 1. / (1. + 2. *λ)#*x
end

"""
    normL2(x::Vector{T}, λ::Float64, kargs...)

returns the value of the L2-regularization term
"""
normL2(x::T, λ::Float64; kargs...) where T<:AbstractArray = λ*norm(x,2)^2
