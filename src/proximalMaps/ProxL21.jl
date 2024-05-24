export L21Regularization

"""
    L21Regularization

Regularization term implementing the proximal map for group-soft-thresholding.

# Arguments
* `λ`                  - regularization paramter

# Keywords
* `slices=1`           - number of elements per group
"""
struct L21Regularization{T} <: AbstractParameterizedRegularization{T}
  λ::T
  slices::Int64
end
L21Regularization(λ; slices::Int64 = 1, kargs...) = L21Regularization(λ, slices)


"""
    prox!(reg::L21Regularization, x, λ)

performs group-soft-thresholding for l1/l2-regularization.
"""
function prox!(reg::L21Regularization, x::Union{AbstractArray{T}, AbstractArray{Complex{T}}},λ::T) where {T}
  return proxL21!(x, λ, reg.slices)
end

function proxL21!(x::Union{AbstractArray{T}, AbstractArray{Complex{T}}}, λ::T, slices::Int64) where T
  sliceLength = div(length(x),slices)
  groupNorm = [norm(x[i:sliceLength:end]) for i=1:sliceLength]
  x[:] = [ x[i]*max( (groupNorm[mod1(i,sliceLength)]-λ)/groupNorm[mod1(i,sliceLength)],0 ) for i=1:length(x)]
  return x
end

"""
    norm(reg::L21Regularization, x, λ)

return the value of the L21-regularization term.
"""
function norm(reg::L21Regularization, x::Union{AbstractArray{T}, AbstractArray{Complex{T}}}, λ::T) where {T}
  sliceLength = div(length(x),reg.slices)
  groupNorm = [norm(x[i:sliceLength:end]) for i=1:sliceLength]
  return λ*norm(groupNorm,1)
end
