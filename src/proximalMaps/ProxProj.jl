export ProjectionRegularization

struct ProjectionRegularization <: AbstractProjectionRegularization
  projFunc::Function
end
ProjectionRegularization(; projFunc::Function=x->x, kargs...) = ProjectionRegularization(projFunc)

function prox!(reg::ProjectionRegularization, x::AbstractArray{Tc}) where {T, Tc <: Union{T, Complex{T}}}
  x[:] = reg.projFunc(x)
  return x
end

function norm(reg::ProjectionRegularization, x::AbstractArray{Tc}) where {T, Tc <: Union{T, Complex{T}}}
  y = copy(x)
  y[:] = prox!(reg, y)
  if y != x
    return Inf
  end
  return 0.
end
