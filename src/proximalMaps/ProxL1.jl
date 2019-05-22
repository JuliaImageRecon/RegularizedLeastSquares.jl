export proxL1!, proxL1, normL1

"""
    proxL1!(x::Array{T}, λ::Float64; sparseTrafo::Trafo=nothing, kargs...) where T

performs soft-thresholding - i.e. proximal map for the Lasso problem.

# Arguments:
* `x::Array{T}`                 - Vector to apply proximal map to
* `λ::Float64`                  - regularization paramter
* `sparseTrafo::Trafo=nothing`  - sparsifying transform to apply
"""
function proxL1!(x::Array{T}, λ::Float64; sparseTrafo::Trafo=nothing, kargs...) where T
  if sparseTrafo != nothing
    z = sparseTrafo*x
  else
    z = x
  end
  if λ != 0
    z[:] = [i*max( (abs(i)-λ)/abs(i),0 ) for i in z]
  end
  if sparseTrafo != nothing
    x[:] = sparseTrafo\z
  else
    x[:] = z
  end
  return x

  # x[:] = [i*max( (abs(i)-λ)/abs(i),0 ) for i in x]
end

"""
    normL1(x::Array{T}, λ::Float64; sparseTrafo::Trafo=nothing, kargs...) where T

returns the value of the L1-regularization term.
Arguments are the same as in `proxL1!`
"""
function normL1(x::Array{T}, λ::Float64; sparseTrafo::Trafo=nothing, kargs...) where T
  if sparseTrafo != nothing
    l1Norm = λ*norm(sparseTrafo*x,1)
  else
    l1Norm = λ*norm(x,1)
  end
  return l1Norm
end
