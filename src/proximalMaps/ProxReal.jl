export RealRegularization, proxReal!, normReal

struct RealRegularization <: AbstractProjectionRegularization
end

"""
    proxReal!(x::Vector{T},λ::Float64=1.0;kargs...) where T

    enforce positivity and realness of solution `x`.
"""
function prox!(::RealRegularization, x::Vector{T}) where T
  enfReal!(x)
end

"""
    returns the value of the characteristic function of real, Real numbers.
    normReal(x) = (isreal(x)&&x>0) ? 0 : Inf
"""
function norm(reg::RealRegularization, x::Vector{T}) where T
  y = copy(x)
  prox!(reg, y)
  if y != x
    return Inf
  end
  return 0
end
