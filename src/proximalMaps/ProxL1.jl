export proxL1!, proxL1, normL1

"""
soft-thresholding for the Lasso problem.
"""
function proxL1!(x::Vector{T}, λ::Float64; kargs...) where T
  x[:] = [i*max( (abs(i)-λ)/abs(i),0 ) for i in x]
end

"""
return the value of the L1-regularization term
"""
normL1(x::Vector{T}, λ::Float64; kargs...) where T = λ*norm(x,1)
