export RealRegularization

"""
    RealRegularization

Regularization term implementing a projection onto real numbers.
"""
struct RealRegularization <: AbstractProjectionRegularization
end

"""
    prox!(reg::RealRegularization, x, λ)

enforce realness of solution `x`.
"""
function prox!(::RealRegularization, x::AbstractArray{T}) where T
  enfReal!(x)
  return x
end

"""
    norm(reg::RealRegularization, x)

returns the value of the characteristic function of real, Real numbers.
"""
function norm(reg::RealRegularization, x::AbstractArray{T}) where T
  y = copy(x)
  prox!(reg, y)
  if y != x
    return Inf
  end
  return 0
end
