export proxL21!, normL21


"""
group-soft-thresholding for l1/l2-regularization.
"""
function proxL21!(x::Vector{T},λ::Float64; sparseTrafo::Trafo=nothing, slices::Int64=1, kargs...) where T
  if sparseTrafo != nothing
    z = sparseTrafo*x
  else
    z = x
  end
  if λ != 0
    proxL21!(z, λ, slices)
  end
  if sparseTrafo != nothing
    x[:] = sparseTrafo\z
  else
    x[:] = z
  end
  return x
end

function proxL21!(x::Vector{T}, λ::Float64, slices::Int64) where T
  sliceLength = div(length(x),slices)
  groupNorm = [norm(x[i:sliceLength:end]) for i=1:sliceLength]
  x[:] = [ x[i]*max( (groupNorm[mod1(i,sliceLength)]-λ)/groupNorm[mod1(i,sliceLength)],0 ) for i=1:length(x)]
end

"""
return the value of the L21-regularization term
"""
function normL21(x::Vector{T}, λ::Float64; sparseTrafo::Trafo=nothing, slices::Int64=1, kargs...) where T
  if sparseTrafo != nothing
    z = sparseTrafo*x
  else
    z = x
  end
  sliceLength = div(length(z),slices)
  groupNorm = [norm(z[i:sliceLength:end]) for i=1:sliceLength]
  return λ*norm(groupNorm,1)
end
