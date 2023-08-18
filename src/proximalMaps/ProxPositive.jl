export PositiveRegularization, proxPositive!, normPositive

struct PositiveRegularization <: AbstractProjectionRegularization
end

"""
    proxPositive!(x::Vector{T},λ::Float64=1.0;kargs...) where T

    enforce positivity and realness of solution `x`.
"""
proxPositive!(x; kargs...) = prox!(PositiveRegularization, x; kargs...)
function prox!(::Type{<:PositiveRegularization}, x::Vector{T}; kargs...) where T
  enfReal!(x)
  enfPos!(x)
end

"""
    returns the value of the characteristic function of real, positive numbers.
    normPositive(x) = (isreal(x)&&x>0) ? 0 : Inf
"""
normPositive(x; kargs...) = norm(PositiveRegularization, x; kargs...)
function norm(::Type{<:PositiveRegularization}, x::Vector{T};kargs...) where T
  y = copy(x)
  proxPositive!(y)
  if y != x
    return Inf
  end
  return 0
end
