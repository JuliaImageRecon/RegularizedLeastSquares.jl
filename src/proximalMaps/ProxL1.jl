export L1Regularization, proxL1!, proxL1, normL1

struct L1Regularization{T} <: AbstractParameterizedRegularization{T}
  λ::T
  L1Regularization(λ::T; kargs...) where T = new{T}(λ)
end

"""
    proxL1!(x::Array{T}, λ::Float64; kargs...) where T

performs soft-thresholding - i.e. proximal map for the Lasso problem.

# Arguments:
* `x::Array{T}`                 - Vector to apply proximal map to
* `λ::Float64`                  - regularization paramter
"""
function prox!(::L1Regularization, x::AbstractArray{Tc}, λ::T) where {T, Tc <: Union{T, Complex{T}}}
  ε = eps(T)
  x .= max.((abs.(x).-λ),0) .* (x.+ε)./(abs.(x).+ε)
  return x
end

"""
    normL1(x::Array{T}, λ::Float64; kargs...) where T

returns the value of the L1-regularization term.
Arguments are the same as in `proxL1!`
"""
function norm(::L1Regularization, x::AbstractArray{Tc}, λ::T) where {T, Tc <: Union{T, Complex{T}}}
  l1Norm = λ*norm(x,1)
  return l1Norm
end
