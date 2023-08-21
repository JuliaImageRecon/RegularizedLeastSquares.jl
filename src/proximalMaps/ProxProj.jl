export ProjectionRegularization, proxProj!, normProj

struct ProjectionRegularization{T} <: AbstractParameterizedRegularization{T}
  λ::T
  projFunc::Function
end
ProjectionRegularization(λ; projFunc::Function=x->x, kargs...) = ProjectionRegularization(λ, projFunc)

"""
    proxProj!(x::Vector{T}, λ::Float64; projFunc=x->x, kargs...)

applies the projection given by `projFunc`.
"""
function prox!(::ProjectionRegularization, x::Vector{Tc}, λ::T; projFunc=x->x, kargs...) where {T, Tc <: Union{T, Complex{T}}}
  x[:] = projFunc(x)
end

"""
    normProj(x::Vector{T}, λ::Float64=0.0; projFunc=x->x, kargs...) where T

evaluate indicator function of set to be projected onto.
"""
function norm(::ProjectionRegularization, x::Vector{Tc}, λ::T=0.0; projFunc=x->x, kargs...) where {T, Tc <: Union{T, Complex{T}}}
  y = copy(x)
  y[:] = proxProj!(y,λ,projFunc=projFunc)
  if y != x
    return Inf
  end
  return 0.
end
