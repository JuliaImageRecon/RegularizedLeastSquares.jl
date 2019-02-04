export proxSLR!, normSLR

"""
proximal map for LLR regularization using singular-value-thresholding

### parameters:

* λ::Float64: regularization parameter (threshold)
* shape::Tuple{Int}: dimensions of the image
* patches::Vector{Vector{Int}}: indices corresponding to each patch
"""
function proxSLR!(x::Vector{T},λ::Float64; shape::NTuple=(1,1), slrIdx=nothing) where T
  for i=1:length(slrIdx)
    svt!(x, shape, λ, slrIdx[i])
  end
end

function svt!(x::Vector{T}, shape::NTuple, λ::Float64, patch::Vector{Int64}) where T

  x = reshape( x, prod(shape), div(length(x),prod(shape)) )

  # threshold singular values
  SVDec = svd(x[patch,:])
  λ_thresh = λ*sqrt(length(patch))
  proxL1!(SVDec.S,λ_thresh)
  x[patch,:] = SVDec.U*Matrix(Diagonal(SVDec.S))*SVDec.Vt
end

"""
return the value of the SLR-regularization term
"""
function normSLR(x::Vector{T},λ::Float64; shape::NTuple=(1,1), slrIdx=nothing) where T
  x = reshape( x, prod(shape), div(length(x),prod(shape)) )
  norm_slr = 0.0
  for i=1:length(slrIdx)
    SVDec = svd(x[slrIdx[i],:])
    norm_slr += λ*sqrt(length(slridx[i]))*norm(SVDec.S,1)
  end

  return norm_slr
end
