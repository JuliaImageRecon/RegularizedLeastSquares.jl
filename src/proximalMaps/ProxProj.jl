export ProjectionRegularization, proxProj!, normProj

struct ProjectionRegularization <: AbstractProjectionRegularization
  projFunc::Function
end
ProjectionRegularization(; projFunc::Function=x->x, kargs...) = ProjectionRegularization(projFunc)

"""
    proxProj!(x::Vector{T}, λ::Float64; projFunc=x->x, kargs...)

applies the projection given by `projFunc`.
"""
function prox!(reg::ProjectionRegularization, x::Vector{Tc}) where {T, Tc <: Union{T, Complex{T}}}
  x[:] = reg.projFunc(x)
end

"""
    normProj(x::Vector{T}, λ::Float64=0.0; projFunc=x->x, kargs...) where T

evaluate indicator function of set to be projected onto.
"""
function norm(reg::ProjectionRegularization, x::Vector{Tc}) where {T, Tc <: Union{T, Complex{T}}}
  y = copy(x)
  y[:] = prox!(reg, y)
  if y != x
    return Inf
  end
  return 0.
end
