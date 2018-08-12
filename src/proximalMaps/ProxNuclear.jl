export proxNuclear!

"""
singular value soft-thresholding.
"""
function proxNuclear!(reg::Regularization, x)
  proxNuclear!(x, reg.params[:lambdNuclear], reg.params[:svtShape])
end

function proxNuclear!(x, λ, shape)
  U,S,V = svd(reshape(x, shape))
  proxL1!(S,λ)
  x[:] = vec(U*diagm(S)*V')
end

function normNuclear(reg::Regularization, x)
  U,S,V = svd( reshape(x, reg.params[:svtShape]) )
  return reg.params[:lambdNuclear]*norm(S,1)
end
