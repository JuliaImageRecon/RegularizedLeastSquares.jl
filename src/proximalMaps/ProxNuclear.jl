export NuclearRegularization


"""
    NuclearRegularization

Regularization term implementing the proximal map for singular value soft-thresholding.

# Arguments:
* `λ`           - regularization paramter

# Keywords
* `svtShape::NTuple`  - size of the underlying matrix
"""
struct NuclearRegularization{T} <: AbstractParameterizedRegularization{T}
  λ::T
  NuclearRegularization(λ::T; kargs...)  where T = new{T}(λ)
end

"""
    prox!(reg::NuclearRegularization, x, λ)

performs singular value soft-thresholding - i.e. the proximal map for the nuclear norm regularization.
"""
function prox!(reg::NuclearRegularization, x::AbstractArray{Tc}, λ::T) where {T, Tc <: Union{T, Complex{T}}}
  U,S,V = svd(x)
  prox!(L1Regularization, S, λ)
  x[:] = vec(U*Matrix(Diagonal(S))*V')
  return x
end

"""
    norm(reg::NuclearRegularization, x, λ)

returns the value of the nuclear norm regularization term.
"""
function norm(reg::NuclearRegularization, x::AbstractArray{Tc}, λ::T) where {T, Tc <: Union{T, Complex{T}}}
  U,S,V = svd(x)
  return λ*norm(S,1)
end
