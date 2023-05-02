"""
    gradOp(T::Type, shape::NTuple{N,Int64})

Nd gradient operator for an array of size `shape`
"""
function GradientOp(T::Type, shape)
  shape = typeof(shape) <: Number ? (shape,) : shape # convert Number to Tuple
  return vcat([GradientOp(T, shape, i) for i ∈ eachindex(shape)]...)
end

"""
    gradOp(T::Type, shape::NTuple{N,Int64}, dims)

directional gradient operator along the dimensions `dims`
for an array of size `shape`
"""
function GradientOp(T::Type, shape::NTuple{N,Int64}, dims) where N
  return vcat([GradientOp(T, shape, dim) for dim ∈ dims]...)
end
function GradientOp(T::Type, shape::NTuple{N,Int64}, dim::Integer) where N
  nrow = div( (shape[dim]-1)*prod(shape), shape[dim] )
  ncol = prod(shape)
  return LinOp{T}(nrow, ncol, false, false,
                          (res,x) -> (grad!(res,x,shape,dim) ),
                          (res,x) -> (grad_t!(res,x,shape,dim) ),
                          nothing )
end

# directional gradients
function grad!(res::T, img::U, shape, dim) where {T<:AbstractVector, U<:AbstractVector}
  img_ = reshape(img,shape)

  δ = zeros(Int, length(shape))
  δ[dim] = 1
  δ = Tuple(δ)
  di = CartesianIndex(δ)

  res_ = reshape(res, shape .- δ)

  Threads.@threads for i ∈ CartesianIndices(res_)
    @inbounds res_[i] = img_[i] - img_[i + di]
  end
end


# adjoint of directional gradients
function grad_t!(res::T, g::U, shape::NTuple{N,Int64}, dim::Int64) where {T<:AbstractVector, U<:AbstractVector, N}
  δ = zeros(Int, length(shape))
  δ[dim] = 1
  δ = Tuple(δ)
  di = CartesianIndex(δ)

  res_ = reshape(res,shape)
  g_ = reshape(g, shape .- δ)

  res_ .= 0
  Threads.@threads for i ∈ CartesianIndices(g_)
    @inbounds res_[i]  = g_[i]
  end
  Threads.@threads for i ∈ CartesianIndices(g_)
    @inbounds res_[i + di] -= g_[i]
  end
end