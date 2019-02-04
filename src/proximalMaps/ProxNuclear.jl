export proxNuclear!

"""
singular value soft-thresholding.
"""
function proxNuclear!(x::Vector{T}, λ::Float64; svtShape::NTuple=[],kargs...) where T
  U,S,V = svd(reshape(x, shape))
  proxL1!(S,λ)
  x[:] = vec(U*diagm(S)*V')
end

function normNuclear(x::Vector{T}, λ::Float64; svtShape::NTuple=[],kargs...) where T
  U,S,V = svd( reshape(x, reg.params[:svtShape]) )
  return reg.params[:lambdNuclear]*norm(S,1)
end
