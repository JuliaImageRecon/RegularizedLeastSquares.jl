export ProjectionRegularization, proxProj!, normProj

struct ProjectionRegularization <: AbstractRegularization
  λ::Float64
  projFunc::Function
end
ProjectionRegularization(λ; projFunc::Function=x->x, kargs...) = ProjectionRegularization(λ, projFunc)

prox!(reg::ProjectionRegularization, x) = proxProj!(x, reg.λ; projFunct = reg.projFunc)
norm(reg::ProjectionRegularization, x) = normProj(x, reg.λ; projFunct = reg.projFunc)


"""
    proxProj!(x::Vector{T}, λ::Float64; projFunc=x->x, kargs...)

applies the projection given by `projFunc`.
"""
function proxProj!(x::Vector{T}, λ::Float64; projFunc=x->x, kargs...) where T
  x[:] = projFunc(x)
end

"""
    normProj(x::Vector{T}, λ::Float64=0.0; projFunc=x->x, kargs...) where T

evaluate indicator function of set to be projected onto.
"""
function normProj(x::Vector{T}, λ::Float64=0.0; projFunc=x->x, kargs...) where T
  y = copy(x)
  y[:] = proxProj!(y,λ,projFunc=projFunc)
  if y != x
    return Inf
  end
  return 0.
end
