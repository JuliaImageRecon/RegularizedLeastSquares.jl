export proxNuclear!, normNuclear

"""
singular value soft-thresholding.
"""
function proxNuclear!(x::Vector{T}, λ::Float64; svtShape::NTuple=[],kargs...) where T
  U,S,V = svd(reshape(x, svtShape))
  proxL1!(S,λ)
  x[:] = vec(U*Matrix(Diagonal(S))*V')
end

function normNuclear(x::Vector{T}, λ::Float64; svtShape::NTuple=[],kargs...) where T
  U,S,V = svd( reshape(x, svtShape) )
  return λ*norm(S,1)
end
