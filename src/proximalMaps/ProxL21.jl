export L21Regularization, proxL21!, normL21

struct L21Regularization{T} <: AbstractParameterizedRegularization{T}
  λ::T
  slices::Int64
end
L21Regularization(λ; slices::Int64 = 1, kargs...) = L21Regularization(λ, slices)


"""
    proxL21!(x::Vector{T},λ::Float64; slices::Int64=1, kargs...)

group-soft-thresholding for l1/l2-regularization.

# Arguments:
* `x::Array{T}`                 - Vector to apply proximal map to
* `λ::Float64`                  - regularization paramter
* `slices::Int64=1`             - number of elements per group
"""
function prox!(reg::L21Regularization, x::AbstractArray{Tc},λ::T) where {T, Tc <: Union{T, Complex{T}}}
  return proxL21!(x, λ, reg.slices)
end

function proxL21!(x::AbstractArray{T}, λ::Float64, slices::Int64) where T
  sliceLength = div(length(x),slices)
  groupNorm = [norm(x[i:sliceLength:end]) for i=1:sliceLength]
  x[:] = [ x[i]*max( (groupNorm[mod1(i,sliceLength)]-λ)/groupNorm[mod1(i,sliceLength)],0 ) for i=1:length(x)]
end

"""
    normL21(x::Vector{T}, λ::Float64; sparseTrafo::Trafo=nothing, slices::Int64=1, kargs...) where T

return the value of the L21-regularization term.
Arguments are the same as in `proxL21!`
"""
function norm(reg::L21Regularization, x::AbstractArray{Tc}, λ::T) where {T, Tc <: Union{T, Complex{T}}}
  sliceLength = div(length(x),reg.slices)
  groupNorm = [norm(x[i:sliceLength:end]) for i=1:sliceLength]
  return λ*norm(groupNorm,1)
end
