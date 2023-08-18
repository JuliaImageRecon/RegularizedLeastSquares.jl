export RealRegularization, proxReal!, normReal

struct RealRegularization <: AbstractProjectionRegularization
end

"""
    proxReal!(x::Vector{T},λ::Float64=1.0;kargs...) where T

    enforce positivity and realness of solution `x`.
"""
proxReal!(x; kargs...) = prox!(RealRegularization, x; kargs...)
function prox!(::Type{<:RealRegularization}, x::Vector{T}; kargs...) where T
  enfReal!(x)
end

"""
    returns the value of the characteristic function of real, Real numbers.
    normReal(x) = (isreal(x)&&x>0) ? 0 : Inf
"""
normReal(x; kargs...) = norm(RealRegularization, x; kargs...)
function norm(::Type{<:RealRegularization}, x::Vector{T};kargs...) where T
  y = copy(x)
  proxReal!(y)
  if y != x
    return Inf
  end
  return 0
end
