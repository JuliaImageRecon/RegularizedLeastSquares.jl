function RegularizedLeastSquares.proxL21!(x::vecT, λ::T, slices::Int64) where {T, vecT <: Union{AbstractGPUVector{T}, AbstractGPUVector{Complex{T}}}}
  sliceLength = div(length(x),slices)
  groupNorm = [norm(x[i:sliceLength:end]) for i=1:sliceLength]

  gpu_call(x, λ, groupNorm, sliceLength) do ctx, x_, λ_, groupNorm_, sliceLength_
    i = @linearidx(x_)
    @inbounds x_[i] = x_[i]*max( (groupNorm_[mod1(i,sliceLength_)]-λ_)/groupNorm_[mod1(i,sliceLength_)],0)
    return nothing
  end
  return x
end