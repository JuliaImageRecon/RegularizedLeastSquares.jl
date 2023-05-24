export PositiveRegularization, proxPositive!, normPositive

struct PositiveRegularization{T} <: AbstractRegularization{T}
  λ::T
end

"""
    proxPositive!(x::Vector{T},λ::Float64=1.0;kargs...) where T

    enforce positivity and realness of solution `x`.
"""
proxPositive!(x, λ::Float64=1.0; kargs...) = prox!(PositiveRegularization, x, λ; kargs...)
function prox!(::Type{<:PositiveRegularization}, x::Vector{T},λ::Float64=1.0;kargs...) where T
  enfReal!(x)
  enfPos!(x)
end

"""
    returns the value of the characteristic function of real, positive numbers.
    normPositive(x) = (isreal(x)&&x>0) ? 0 : Inf
"""
normPositive(x, λ::Float64=1.0; kargs...) = norm(PositiveRegularization, x, λ; kargs...)
function norm(::Type{<:PositiveRegularization}, x::Vector{T},λ::Float64=1.0;kargs...) where T
  y = copy(x)
  proxPositive!(y)
  if y != x
    return Inf
  end
  return 0.
end
