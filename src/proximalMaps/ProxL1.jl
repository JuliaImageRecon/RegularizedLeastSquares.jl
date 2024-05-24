export L1Regularization

"""
    L1Regularization

Regularization term implementing the proximal map for the Lasso problem.
"""
struct L1Regularization{T} <: AbstractParameterizedRegularization{T}
  λ::T
  L1Regularization(λ::T; kargs...) where T = new{T}(λ)
end

"""
    prox!(reg::L1Regularization, x, λ)

performs soft-thresholding - i.e. proximal map for the Lasso problem.
"""
function prox!(::L1Regularization, x::Union{AbstractArray{T}, AbstractArray{Complex{T}}}, λ::T) where {T}
  ε = eps(T)
  x .= max.((abs.(x).-λ),0) .* (x.+ε)./(abs.(x).+ε)
  return x
end

"""
    norm(reg::L1Regularization, x, λ)

returns the value of the L1-regularization term.
"""
function norm(::L1Regularization, x::Union{AbstractArray{T}, AbstractArray{Complex{T}}}, λ::T) where {T}
  l1Norm = λ*norm(x,1)
  return l1Norm
end
