export proxL21!


"""
group-soft-thresholding for l1/l2-regularization.
"""
function proxL21!(x::Vector{T},λ::Float64; slices::Int64=1, kargs...) where T
  proxL21!(x, λ, slices)
end

function proxL21!(x::Vector{T}, λ::Float64, slices::Int64=1) where T
  sliceLength = div(length(x),slices)
  groupNorm = [norm(x[i:sliceLength:end]) for i=1:sliceLength]
  x[:] = [ x[i]*max( (groupNorm[mod1(i,sliceLength)]-λ)/groupNorm[mod1(i,sliceLength)],0 ) for i=1:length(x)]
end

"""
return the value of the L21-regularization term
"""
function normL21(x::Vector{T}, λ::Float64; slices::Int64=1, kargs...) where T
  sliceLength = div(length(x),slices)
  groupNorm = [norm(x[i:sliceLength:end]) for i=1:sliceLength]
  return λ*norm(groupNorm,1)
end
