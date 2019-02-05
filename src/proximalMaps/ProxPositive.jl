export proxPositive!, normPositive

"""
enforce positivity and realness of solution.
"""
function proxPositive!(x::Vector{T},λ::Float64=1.0;kargs...) where T
  enfReal!(x)
  enfPos!(x)
end

function normPositive(x::Vector{T},λ::Float64=1.0;kargs...) where T
  y = copy(x)
  proxPositive!(y)
  if y != x
    return Inf
  end
  return 0.
end
