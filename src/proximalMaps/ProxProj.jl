export ProjectionRegularization, proxProj!, normProj

struct ProjectionRegularization{T} <: AbstractRegularization{T}
  λ::T
  projFunc::Function
end
ProjectionRegularization(λ; projFunc::Function=x->x, kargs...) = ProjectionRegularization(λ, projFunc)

"""
    proxProj!(x::Vector{T}, λ::Float64; projFunc=x->x, kargs...)

applies the projection given by `projFunc`.
"""
proxProj!(x, λ; kargs...) = prox!(ProjectionRegularization, x, λ; kargs...)
function prox!(::Type{<:ProjectionRegularization}, x::Vector{T}, λ::Float64; projFunc=x->x, kargs...) where T
  x[:] = projFunc(x)
end

"""
    normProj(x::Vector{T}, λ::Float64=0.0; projFunc=x->x, kargs...) where T

evaluate indicator function of set to be projected onto.
"""
normProj(x, λ=0.0; kargs...) = norm(ProjectionRegularization, x, λ; kargs...)
function norm(::Type{<:ProjectionRegularization}, x::Vector{T}, λ::Float64=0.0; projFunc=x->x, kargs...) where T
  y = copy(x)
  y[:] = proxProj!(y,λ,projFunc=projFunc)
  if y != x
    return Inf
  end
  return 0.
end
