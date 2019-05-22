export proxL2!, normL2


"""
    proxL2!(x::Vector{T}, λ::Float64; kargs...) where T

proximal map for Tikhonov regularization.
"""
function proxL2!(x::Vector{T}, λ::Float64; kargs...) where T
  x[:] = 1. / (1. + 2. *λ)*x
end

"""
    normL2(x::Vector{T}, λ::Float64, kargs...)

returns the value of the L2-regularization term
"""
normL2(x::Vector{T}, λ::Float64, kargs...) where T = λ*norm(x,2)^2
