"""
    gradOp(T::Type, shape::NTuple{1,Int64})

1d gradient operator for an array of size `shape`
"""
GradientOp(T::Type, shape::NTuple{1,Int64}) = GradientOp(T,shape,1)

"""
    gradOp(T::Type, shape::NTuple{2,Int64})

2d gradient operator for an array of size `shape`
"""
function GradientOp(T::Type, shape::NTuple{2,Int64})
  return vcat( GradientOp(T,shape,1), GradientOp(T,shape,2) ) 
end

"""
    gradOp(T::Type, shape::NTuple{3,Int64})

3d gradient operator for an array of size `shape`
"""
function GradientOp(T::Type, shape::NTuple{3,Int64})
  return vcat( GradientOp(T,shape,1), GradientOp(T,shape,2), GradientOp(T,shape,3) ) 
end

"""
    gradOp(T::Type, shape::NTuple{N,Int64}, dim::Int64) where N

directional gradient operator along the dimension `dim`
for an array of size `shape`
"""
function GradientOp(T::Type, shape::NTuple{N,Int64}, dim::Int64) where N
  nrow = div( (shape[dim]-1)*prod(shape), shape[dim] )
  ncol = prod(shape)
  return LinOp{T}(nrow, ncol, false, false,
                          (res,x) -> (grad!(res,x,shape,dim) ), 
                          (res,x) -> (grad_t!(res,x,shape,dim) ), 
                          nothing )
end

# directional gradients
function grad!(res::T, img::U, shape::NTuple{1,Int64}, dim::Int64) where {T<:AbstractVector,U<:AbstractVector}
  res .= img[1:end-1].-img[2:end]
end

function grad!(res::T, img::U, shape::NTuple{2,Int64}, dim::Int64) where {T<:AbstractVector,U<:AbstractVector}
  img = reshape(img,shape)

  if dim==1
    # res .= vec(img[1:end-1,:].-img[2:end,:])
    res .= vec(img[1:end-1,:])
    res .-= vec(img[2:end,:])
  else
    # res .= vec(img[:,1:end-1].-img[:,2:end])
    res .= vec(img[:,1:end-1])
    res .-= vec(img[:,2:end])
  end
end

function grad!(res::T,img::U, shape::NTuple{3,Int64}, dim::Int64) where {T<:AbstractVector,U<:AbstractVector}
  img = reshape(img,shape)

  if dim==1
    # res .= vec(img[1:end-1,:,:].-img[2:end,:,:])
    res .= vec(img[1:end-1,:,:])
    res .-= vec(img[2:end,:,:])
  elseif dim==2
    # res.= vec(img[:,1:end-1,:].-img[:,2:end,:])
    res .= vec(img[:,1:end-1,:])
    res .-= vec(img[:,2:end,:])
  else
    # res.= vec(img[:,:,1:end-1].-img[:,:,2:end])
    res .= vec(img[:,:,1:end-1])
    res .-= vec(img[:,:,2:end])
  end
end

# adjoint of directional gradients
function grad_t!(res::T, g::U, shape::NTuple{1,Int64}, dim::Int64) where {T<:AbstractVector,U<:AbstractVector}
  res .= zero(eltype(g))
  res[1:shape[1]-1] .= g
  res[2:shape[1]] .-= g
end

function grad_t!(res::T, g::U, shape::NTuple{2,Int64}, dim::Int64) where {T<:AbstractVector,U<:AbstractVector}
  res .= zero(eltype(g))
  res_ = reshape(res,shape)

  if dim==1
    g = reshape(g,shape[1]-1,shape[2])
    res_[1:shape[1]-1,:] .= g
    res_[2:shape[1],:] .-= g
  else
    g = reshape(g,shape[1],shape[2]-1)
    res_[:,1:shape[2]-1] .= g
    res_[:,2:shape[2]] .-= g
  end
end

function grad_t!(res::T, g::U, shape::NTuple{3,Int64}, dim::Int64) where {T<:AbstractVector,U<:AbstractVector}
  res .= zero(eltype(g))
  res_ = reshape(res,shape)

  if dim==1
    g = reshape(g,shape[1]-1,shape[2],shape[3])
    res_[1:shape[1]-1,:,:] .= g
    res_[2:shape[1],:,:] .-= g
  elseif dim==2
    g = reshape(g,shape[1],shape[2]-1,shape[3])
    res_[:,1:shape[2]-1,:] .= g
    res_[:,2:shape[2],:] .-= g
  else
    g = reshape(g,shape[1],shape[2],shape[3]-1)
    res_[:,:,1:shape[3]-1] .= g
    res_[:,:,2:shape[3]] .-= g
  end
end