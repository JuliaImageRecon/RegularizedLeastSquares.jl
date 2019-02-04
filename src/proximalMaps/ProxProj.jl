export proxProj!, normProj

"""
projection operator.
"""
function proxProj!(x::Vector{T}, λ::Float64; projFunc=x->x, kargs...) where T
  projFunc!(x)
end

"""
evaluate indicator function of set to be projected onto
"""
function normProj(x::Vector{T}, λ::Float64; projFunc=x->x, kargs...) where T
  y = copy(x)
  proxProj!(y)
  if y != x
    return Inf
  end
  return 0.
end
