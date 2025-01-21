function RegularizedLeastSquares.proxL21!(x::vecT, λ::T, slices::Int64) where {T, vecT <: Union{AbstractGPUVector{T}, AbstractGPUVector{Complex{T}}}}
  sliceLength = div(length(x),slices)
  groupNorm = copyto!(similar(x, Float32, sliceLength), [Float32(norm(x[i:sliceLength:end])) for i=1:sliceLength])

  @kernel inbounds = true cpu = false function proxL21_kernel(x, λ, groupNorm, sliceLength)
    i = @index(Global, Linear)
    x[i] = x[i]*max( (groupNorm[mod1(i,sliceLength)]-λ)/groupNorm[mod1(i,sliceLength)],0)
  end
  kernel! = proxL21_kernel(get_backend(x))
  kernel!(x, λ, groupNorm, sliceLength, ndrange = length(x))
  return x
end