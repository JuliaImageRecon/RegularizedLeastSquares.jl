export proxProj!, normProj

"""
projection operator.
"""
function proxProj!(reg::Regularization, x)
  projFunc! = get(reg.params, :projFunc, x->x)
  projFunc!(x)
end

"""
evaluate indicator function of set to be projected onto
"""
function normProj(reg::Regularization,x)
  y = copy(x)
  proxProj!(y)
  if y != x
    return Inf
  end
  return 0.
end
