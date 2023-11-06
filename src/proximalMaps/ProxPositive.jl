export PositiveRegularization

"""
    PositiveRegularization

Regularization term implementing a projection onto positive and real numbers.
"""
struct PositiveRegularization <: AbstractProjectionRegularization
end

"""
    proxPositive!(reg::PositiveRegularization, x) where T

enforce positivity and realness of solution `x`.
"""
function prox!(::PositiveRegularization, x::Vector{T}) where T
  enfReal!(x)
  enfPos!(x)
  return x
end

"""
    norm(reg::PositiveRegularization, x)

returns the value of the characteristic function of real, positive numbers.
"""
function norm(reg::PositiveRegularization, x::Vector{T}) where T
  y = copy(x)
  prox!(reg, y)
  if y != x
    return Inf
  end
  return 0
end
