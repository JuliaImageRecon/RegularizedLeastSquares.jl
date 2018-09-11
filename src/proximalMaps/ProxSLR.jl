export proxSLR!, normSLR

"""
proximal map for LLR regularization using singular-value-thresholding

### parameters:

* λ::Float64: regularization parameter (threshold)
* shape::Tuple{Int}: dimensions of the image
* patches::Vector{Vector{Int}}: indices corresponding to each patch
"""
function proxSLR!(reg, x)
  patchIdx = reg.params[:slrIdx]
  for i=1:length(patchIdx)
    svt!(x, reg.params[:shape], reg.params[:lambdSLR], patchIdx[i])
  end
end

function svt!(x::Vector{T}, shape::Tuple, λ::Float64, patch::Vector{Int64}) where T

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
function normSLR(reg::Regularization,x)
  patchIdx = reg.params[:slrIdx]
  shape = reg.params[:shape]
  x = reshape( x, prod(shape), div(length(x),prod(shape)) )
  λ = reg.params[:lambdSLR]

  norm_slr = 0.0
  for i=1:length(patchIdx)
    SVDec = svd(x[patchIdx[i],:])
    norm_slr += λ*sqrt(length(patchIdx[i]))*norm(SVDec.S,1)
  end

  return norm_slr
end
