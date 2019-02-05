export proxL2!, normL2


"""
proximal map for Tikhonov regularization.
"""
function proxL2!(x::Vector{T}, λ::Float64; kargs...) where T
  x[:] = 1. / (1. + 2. *λ)*x
end

"""
return the value of the L2-regularization term
"""
normL2(x::Vector{T}, λ::Float64, kargs...) where T = λ*norm(x,2)^2
