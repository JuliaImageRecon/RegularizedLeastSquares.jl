export PositiveRegularization, proxPositive!, normPositive

struct PositiveRegularization <: AbstractProjectionRegularization
end

"""
    proxPositive!(x::Vector{T},λ::Float64=1.0;kargs...) where T

    enforce positivity and realness of solution `x`.
"""
function prox!(::PositiveRegularization, x::Vector{T}) where T
  enfReal!(x)
  enfPos!(x)
end

"""
    returns the value of the characteristic function of real, positive numbers.
    normPositive(x) = (isreal(x)&&x>0) ? 0 : Inf
"""
function norm(reg::PositiveRegularization, x::Vector{T}) where T
  y = copy(x)
  prox!(reg, y)
  if y != x
    return Inf
  end
  return 0
end
