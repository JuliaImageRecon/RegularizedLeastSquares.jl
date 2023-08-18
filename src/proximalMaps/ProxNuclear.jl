export NuclearRegularization, proxNuclear!, normNuclear

struct NuclearRegularization{T} <: AbstractParameterizedRegularization{T}
  λ::T
  svtShape::NTuple
end
NuclearRegularization(λ; svtShape::NTuple=[], kargs...) = NuclearRegularization(λ, svtShape)

"""
    proxNuclear!(x::Vector{T}, λ::Float64; svtShape::NTuple=[],kargs...)

applies singular value soft-thresholding - i.e. the proximal map for the nuclear norm regularization.

# Arguments:
* `x::Array{T}`          - Vector to apply proximal map to
* `λ::Float64`           - regularization paramter
* `svtShape::NTuple=[]`  - size of the underlying matrix
"""
proxNuclear!(x, λ; kargs...) = prox!(NuclearRegularization, x, λ; kargs...)
function prox!(::Type{<:NuclearRegularization}, x::Vector{T}, λ::Float64; svtShape::NTuple=[],kargs...) where T
  U,S,V = svd(reshape(x, svtShape))
  proxL1!(S,λ)
  x[:] = vec(U*Matrix(Diagonal(S))*V')
end

"""
    normNuclear(x::Vector{T}, λ::Float64; svtShape::NTuple=[],kargs...) where T

returns the value of the nuclear norm regularization term.
Arguments are the same as in `proxNuclear!`
"""
normNuclear(x, λ; kargs...) = norm(NuclearRegularization, x, λ; kargs...)
function norm(::Type{<:NuclearRegularization}, x::Vector{T}, λ::Float64; svtShape::NTuple=[],kargs...) where T
  U,S,V = svd( reshape(x, svtShape) )
  return λ*norm(S,1)
end
