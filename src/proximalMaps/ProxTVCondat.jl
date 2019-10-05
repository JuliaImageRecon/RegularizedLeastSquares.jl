export proxTVCondat!


function proxTVCondat!(x::Vector{T}, λ::Float64; shape=[], kargs...) where T
  x_ = reshape(x, shape...)
  nhood = Int64[1 0 0;0 1 0; 0 0 1]
  omega = T[1,1,1]

  y = copy(x_)
  y[:] .= 0.0

  for d=1:length(omega)
    y_ = copy(x_)
    tv_denoise_3d_condat!(y_, nhood[d,:], λ*omega[d])
    y .+= y_ ./ length(omega)
  end
  x[:] = y[:]
  return y
end

mutable struct StartRange <: AbstractArray{CartesianIndices,3}
  x::CartesianIndices
  y::CartesianIndices
  z::CartesianIndices
end

Base.size(R::StartRange) = (Int(3),)
Base.IndexStyle(::Type{StartRange}) = IndexLinear()
Base.getindex(R::StartRange,i::Int) = if i==1 R.x elseif i==2 R.y elseif i==3 R.z end;

"""
This function checks if the cartesian index exceeds size.
"""
function inrange(size::Tuple{Int64,Int64,Int64},range::CartesianIndex{3})

    if range.I[1] > size[1] || range.I[1] < 1 || range.I[2] > size[2] ||
       range.I[2] < 1 || range.I[3] > size[3] || range.I[3] < 1
      return false
    else
      return true
    end

end


"""
This function returns a StartRange variable, which contains the start planes for
the 1d tv extraction.
"""
function get_startrange(size::Tuple{Int64,Int64,Int64},step::Array{T,1}) where {T<:Real}

  output = StartRange(
    CartesianIndices(CartesianIndex((0,0,0))),
    CartesianIndices(CartesianIndex((0,0,0))),
    CartesianIndices(CartesianIndex((0,0,0)))
  )

  output.x = CartesianIndices((1:step[1],1:size[2],1:size[3]))
  output.y = CartesianIndices((1:size[1],1:step[2],1:size[3]))
  output.z = CartesianIndices((1:size[1],1:size[2],1:step[3]))

  if step[1] < 0
    output.x = CartesianIndices(((size[1]+step[1]):size[1],1:size[2],1:size[3]))
  end

  if step[2] < 0
    output.y = CartesianIndices((1:size[1],(size[2]+step[2]):size[2],1:size[3]))
  end

  if step[3] < 0
    output.z = CartesianIndices((1:size[1],1:size[2],(size[3]+step[3]):size[3]))
  end

  return output
end


"""
This function returns the one dimensional tv problem depending on a start pixel
neighbor and a direction increment.
"""
function tv_get_onedim_data!(tvData::Array{T,3}, tvOneDim::Array{T,1}, neighbor,
    increment, arrayCount::Array{Int64,1}) where {T<:Real}

  tvSize = size(tvData)

  arrayCount[1] = 0

  # While neighbor does not exceeds tvSize, add data to 1d problem
  while true
    if inrange(tvSize,neighbor)
      arrayCount[1] = arrayCount[1]+1
      @inbounds tvOneDim[arrayCount[1]] = tvData[neighbor]
      neighbor = neighbor + increment
    else
      break
    end
  end

end


"""
This function sorts the 1d tv result back into the 3d data.
"""
function tv_push_onedim_data!(tvData::Array{T,3},tvOneDim::Array{T,1},
     arrayCount::Int64,neighbor,increment) where {T<:Real}

  for i=1:arrayCount
    @inbounds tvData[neighbor] = tvOneDim[i]
    neighbor = neighbor + increment
  end

end

"""
This function extracts 1d problems from the 3d data and starts the 1d tv function.
"""
function tv_denoise_3d_condat!(tvData::Array{T,3}, nhood::Array{Int64,1}, lambda) where {T<:Real}

  tvSize = size(tvData)
  cartRange = get_startrange(tvSize,nhood[:])
  increment = CartesianIndex((nhood[1],nhood[2],nhood[3]));
  tvOneDim = Array{eltype(tvData)}(undef, Int64(ceil(sqrt(tvSize[1]*tvSize[2]*tvSize[3]))))
  arrayCount = Array{Int64}(undef, 1)
  for R in cartRange

    for k in R
      neighbor = k;
      tv_get_onedim_data!(tvData,tvOneDim,neighbor,increment,arrayCount)

      tv_denoise_1d_condat!(tvOneDim,arrayCount[1],lambda)

      neighbor = k

      tv_push_onedim_data!(tvData,tvOneDim,arrayCount[1],neighbor,increment)
    end
  end
end


"""
This function performs the 1d tv algorithm.
"""
function tv_denoise_1d_condat!(c::Array{T,1},width::Int64,lambda) where {T<:Real}

  cLength = width

  k = 1
  k0 = 1
  umin = lambda
  umax = -lambda
  vmin = c[1] - lambda
  vmax = c[1] + lambda
  kplus = 1
  kminus = 1

  twolambda = 2*lambda
  minlambda = -lambda

  while true

    while k == cLength
      if umin < 0
        while true
          c[k0] = vmin
          k0 = k0+1
          !(k0<=kminus) && break
        end

        k=k0
        kminus=k
        vmin=c[kminus]
        umin=lambda

        umax = vmin + umin - vmax
      elseif umax > 0
        while true
          c[k0] = vmax
          k0 = k0+1
          !(k0<=kplus) && break
        end

        k=k0
        kplus=k
        vmax=c[kplus]
        umax=minlambda

        umin = vmax + umax - vmin
      else
          vmin = vmin + umin/(k-k0+1)
          while true
            c[k0] = vmin
            k0 = k0 + 1
            !(k0<=k) && break
          end
          return
      end
    end

    umin = umin + c[k+1] - vmin
    if umin < minlambda
      # Inplace soft thresholding
      #vmin = vmin>mu ? vmin-mu : vmin<-mu ? vmin+mu : 0.0
      while true
        c[k0] = vmin
        k0 = k0 + 1
        !(k0<=kminus) && break
      end
      k=k0
      kminus=k
      kplus=kminus
      vmin = c[kplus]
      vmax = vmin + twolambda
      umin = lambda
      umax = minlambda
    else
      umax = umax + c[k+1] - vmax
      if umax > lambda
        # Inplace soft thresholding
        #vmax = vmax>mu ? vmax-mu : vmax<-mu ? vmax+mu : 0.0;
        while true
          c[k0]=vmax
          k0 = k0 + 1
          !(k0<=kplus) && break
        end
        k = k0
        kminus = k
        kplus = kminus
        vmax = c[kplus]
        vmin = vmax - twolambda
        umin = lambda
        umax = minlambda
      else
        k = k + 1
        if umin >= lambda
          kminus = k
          vmin = vmin + (umin-lambda)/(kminus-k0+1)
          umin = lambda
        end
        if umax <= minlambda
          kplus = k
          vmax = vmax + (umax+lambda)/(kplus -k0 +1)
          umax = minlambda
        end
      end
    end
  end
end
