export FusedLasso
export fusedlasso

mutable struct FusedLasso{T} <: AbstractLinearSolver
    # System matrix
    A::Array{T,2}
    # Regularizer
    reg::Vector{Regularization}
    # Solution
    x::Vector{T}
    # Measurement vector
    b::Vector{T}
    # 3D dimensions of solution
    shape::Array{Int64,1}
    # Row energies of system matrix
    rowEnergy
    # user params
    iterations::Int64
    nhood::Array{Int64,2}
    omega::Vector{T}
    gamma::T
    lambda::T
end

function FusedLasso(S::Matrix{T};
                    λ=[0.0002,0.0016],
                    shape::Array{Int64,1}=[size(S,2),1,1],
                    iterations = Int64(50),
                    directions = 3,
                    kappa = T(0.5),
                    gamma = T(10.0^-3),
                    kargs...) where T

    reg = Regularization(["TV","L1"], λ)

    # Determine direction vectors and weights for proximal mapping
  	if directions == 3
              nhood = Int64[1 0 0;0 1 0; 0 0 1]
              omega = T[1,1,1]

  	elseif directions == 13
  		 nhood = Int64[
  		    1 0 0;
  		    0 1 0;
  		    0 0 1;
  		    1 1 0;
  		    1 -1 0;
  		    1 0 1;
  		    1 0 -1;
  		    0 1 1;
  		    0 1 -1;
  		    1 1 1;
  		    1 1 -1;
  		    1 -1 -1;
  		    -1 1 -1
  		   ]
  		omega = T[
  		(2/sqrt(3) - 1)
  		(2/sqrt(3) - 1)
  		(2/sqrt(3) - 1)
  		((3 * sqrt(2) - 2 * sqrt(3)) / 6)
  		((3 * sqrt(2) - 2 * sqrt(3)) / 6)
  		((3 * sqrt(2) - 2 * sqrt(3)) / 6)
  		((3 * sqrt(2) - 2 * sqrt(3)) / 6)
  		((3 * sqrt(2) - 2 * sqrt(3)) / 6)
  		((3 * sqrt(2) - 2 * sqrt(3)) / 6)
  		((3 - 3 * sqrt(2) + sqrt(3)) / 6)
  		((3 - 3 * sqrt(2) + sqrt(3)) / 6)
  		((3 - 3 * sqrt(2) + sqrt(3)) / 6)
  		((3 - 3 * sqrt(2) + sqrt(3)) / 6)
  		]
    else
      @warn "No corresponding neighbourhood for direction number. 3 directions will be used!"
      nhood = Int64[1 0 0;0 1 0; 0 0 1]
      omega = T[1,1,1]
  	end

    # Call constructor interface (function below) and get linear solver object
    solver = FusedLasso(copy(S), reg,
               zeros(eltype(S),shape[1]*shape[2]*shape[3]),
               zeros(eltype(S), div(length(S),shape[1]*shape[2]*shape[3])),shape,
               zeros(eltype(S),size(S,1)),
               iterations,nhood,omega,gamma,kappa
            )

   # Store row energies and normalize rows of system matrix
   normalize!(solver.A, solver.b, solver.rowEnergy)

   return solver
end


#==============================================================================#
# Parameter Settings
#==============================================================================#

function normalize!(A::Array{T,2},b::Vector{T},energy::Vector{T}) where {T<:Real}

    for i=1:size(A,1)
      energy[i] = norm(vec(A[i,:]));
      for j=1:size(A,2)
        A[i,j] = A[i,j]/energy[i]
      end
      b[i] = b[i]/energy[i]
    end

end


function normalize!(b::Vector{T},energy::Vector{T}) where T
    for i=1:length(energy)
      b[i] = b[i] / energy[i]
    end
end

function setMeasurement!(solver::FusedLasso, measurement::Vector)
   m = copy(measurement)
   # Multiply measurement vector with row energies of system matrix
   normalize!(m,solver.rowEnergy)
   # Store result
   solver.b = m
end

#==============================================================================#
# Solve interface
#==============================================================================#

function solve(solver::FusedLasso, measurement::Vector)
    setMeasurement!(solver, measurement)
    solve(solver)
end

function solve(solver::FusedLasso{T}) where T
      shape = solver.shape
      solver.x[:] .= 0.0
      #linearProblem.x = zeros(eltype(linearProblem.x),shape[1]*shape[2]*shape[3])
      solver.x = vec(fusedlasso(
                reshape(solver.A ,size(solver.A,1),shape[1],shape[2],shape[3]),
                solver.b,
                reshape(solver.x,shape[1],shape[2],shape[3]);
                iterations = solver.iterations,
                nhood = solver.nhood,
                omega = solver.omega,
                lambda = solver.lambda,
                alpha = T(solver.reg[1].λ),
                beta = T(solver.reg[2].λ),
                gamma = solver.gamma)   )
end

#==============================================================================#
# Algorithm
#==============================================================================#

mutable struct StartRange <: AbstractArray{CartesianIndices,3}
  x::CartesianIndices
  y::CartesianIndices
  z::CartesianIndices
end

Base.size(R::StartRange) = (Int(3),)
Base.IndexStyle(::Type{StartRange}) = IndexLinear()
Base.getindex(R::StartRange,i::Int) = if i==1 R.x elseif i==2 R.y elseif i==3 R.z end;

"""This function implements the base version of fused lasso reconstruction.

### Keyword/Optional Arguments
* 'iterations::Int64' Maximum number of Iterations.
* 'tol::Float32' Tolerance for the stopping criterion.
* 'nhood::Array{Int64,N,3}' Neighborhood relationships for tv problem.
* 'omega::Array{Float32,N}' Vector with weights for the tv results.
* 'lambda::Float32' Scale for the update step.
* 'alpha::Float32' Weight of the tv term.
* 'beta::Float32' Weight of the l1 term.
* 'gamma::Float32' Weight for the gradient descent step.
"""
function fusedlasso(S::Array{T,4},u::Vector{T},c::Array{T,3};
  iterations = Int64(50),
  tol = T(5.0*10.0^-6),
  nhood = Int64[1 0 0;0 1 0; 0 0 1],
  omega = T[1 1 1],
  lambda = T(1),
  alpha = T(0.000017),
  beta = T(0.001),
  gamma = T(10.0^-3),
  kargs...
  ) where T
  gradientFunc! = gradient_base!

  cSize = size(c)
  N = size(collect(omega),1)
  z = zeros(eltype(c),N+1,cSize[1],cSize[2],cSize[3])

  y = zeros(eltype(c),cSize[1],cSize[2],cSize[3])

  t = ones(T,N+1)/(N+1)
  yTemp = Array{eltype(y)}(undef, size(y))
  #cOld = copy(c)
  zTemp = Array{eltype(yTemp)}(undef, size(yTemp))

  @debug "Residuum pre reconstruction: " residuum(S,c,u)
  for i=1:iterations

      # Calculate gradient
      gradientFunc!(c,S,u,y,gamma)

      # Proximal mapping
      proxmap!(y,yTemp,z,c,N,nhood,alpha,beta,gamma,omega,lambda,t)

      # Pointwise maximum
      pointwise_max!(c,z,N+1,y,yTemp)

      # Update step
      update!(z,N+1,yTemp,c,lambda)

      # Averaging over single results
      c = reshape(c,cSize[1],cSize[2],cSize[3])
      weighted_sum!(t,z,c)
      # Evaluate stopping criterion
      #check_stop(cOld,c,tol) && break;
      #cOld = copy(c)

  end
  @debug "Residuum post reconstruction: " residuum(S,c,u)
  return c

end

"""
This function calculates the error gradient according to y = γA*(Ac-u).
"""
function gradient_base!(c::Array{T,3},A::Array{T,4},u::Array{T,1},y::Array{T,3},α::T) where {T}

  aSize = size(A)
  cSize = size(c)
  uSize = size(u)
  ySize = size(y)

  A = reshape(A,aSize[1],aSize[2]*aSize[3]*aSize[4])
  c = reshape(c,cSize[1]*cSize[2]*cSize[3])
  y = reshape(y,ySize[1]*ySize[2]*ySize[3])

  # Update yTemp with Ac-u
  yTemp = copy(u)
  BLAS.gemv!('N',one(T),A,c,-one(T),yTemp);

  # Update y with γA'yTemp
  BLAS.gemv!('T',α,A,yTemp,zero(T),y);

end

"""
This function performs the proximal mapping.
"""
function proxmap!(y::Array{T,3},yTemp::Array{T,3},z::Array{T,4},c::Array{T,3},
    N,nhood,alpha,beta,gamma,omega,lambda,t) where {T<:Real}
  cSize = size(c)
  ySize = size(y)
  zSize = size(z)

  updateTemp = Array{eltype(y)}(undef, ySize[1]*ySize[2]*ySize[3])

  c = reshape(c,cSize[1]*cSize[2]*cSize[3])
  y = reshape(y,ySize[1]*ySize[2]*ySize[3])
  yTemp = reshape(yTemp,ySize[1]*ySize[2]*ySize[3])

  for s=1:N

      yTemp = 2*c-reshape(z[s,:,:,:],ySize[1]*ySize[2]*ySize[3])-y;

      # Solve TV problems
      yTemp = reshape(yTemp,ySize[1],ySize[2],ySize[3])
      tv_denoise_3d_condat!(yTemp, nhood[s,:], gamma*omega[s]*alpha/t[s])

      # Soft thresholding
      softthresh!(yTemp,gamma * beta/(N*t[s]))

      # Update step
      c = reshape(c,cSize[1],cSize[2],cSize[3])
      update!(z,s,yTemp,c,lambda)
      c = reshape(c,cSize[1]*cSize[2]*cSize[3])

  end
end

"""
This function returns the one dimensional tv problem depending on a start pixel
neighbor and a direction increment.
"""
function tv_get_onedim_data!(tvData::Array{T,3},tvOneDim::Array{T,1},neighbor,
    increment,arrayCount::Array{Int64,1}) where {T<:Real}

  tvSize = size(tvData)

  arrayCount[1] = 0

  # While neighbor does not exceeds tvSize, add data to 1d problem
  while true
    if inrange(tvSize,neighbor)
      arrayCount[1] = arrayCount[1]+1
      @inbounds tvOneDim[arrayCount[1]]=tvData[neighbor]
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
function tv_denoise_3d_condat!(tvData::Array{T,3},nhood::Array{Int64,1},lambda) where {T<:Real}

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


"""
This function applies soft thresholding on given data y.
"""
function softthresh!(y::Array{T,3}, threshold) where {T<:Real}

  ySize = size(y)
  y = reshape(y,ySize[1]*ySize[2]*ySize[3])

  y[:] = [sign(i)*max(i-threshold,0) for i in y]

end


"""
This function performs the update step.
"""
function update!(z::Array{T,4}, s::Int64, y::Array{T,3}, c::Array{T,3}, lambda) where {T<:Real}
  dataSize = size(c)
  flatSize = dataSize[1]*dataSize[2]*dataSize[3];
  y = reshape(y,flatSize)
  c = reshape(c,flatSize)
  yTemp = Array{T}(undef, flatSize)
  zTemp = view(reshape(z,Val(2)),s,:)

  broadcast!(-,yTemp,y,c)

  #yTemp[:] = y[:] - c[:]
  rmul!(yTemp,lambda)
  #zTemp[:] = zTemp[:] + yTemp[:]
  broadcast!(+,zTemp,zTemp,yTemp)
  #z[s,:,:,:] = reshape(yTemp,dataSize)[:,:,:]

end


"""
This function checks if the cartisian index exceeds size.
"""
function inrange(size::Tuple{Int64,Int64,Int64},range::CartesianIndex{3})

    if range.I[1] > size[1] || range.I[1] < 1 || range.I[2] > size[2] || range.I[2] < 1 || range.I[3] > size[3] || range.I[3] < 1
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


  #output.x = CartesianIndices(CartesianIndex((1,1,1)),CartesianIndex((step[1],size[2],size[3])))
  #output.y = CartesianIndices(CartesianIndex((1,1,1)),CartesianIndex((size[1],step[2],size[3])))
  #output.z = CartesianIndices(CartesianIndex((1,1,1)),CartesianIndex((size[1],size[2],step[3])))
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
This function applies the pointwise maximum step.
"""
function pointwise_max!(c::Array{T,3},z::Array{T,4},s::Int64,y::Array{T,3},out::Array{T,3}) where {T<:Real}
  ySize = size(y)
  cSize = size(c)

  y = reshape(y,ySize[1]*ySize[2]*ySize[3])
  c = reshape(c,cSize[1]*cSize[2]*cSize[3])
  zTemp = view(reshape(z,Val(2)),s,:)
  out = reshape(out,cSize[1]*cSize[2]*cSize[3])

  yTemp = 2*c-zTemp-y;

  for j=1:length(yTemp)
    out[j] = max(yTemp[j],0)
  end

end


"""
This function calculates the weighted sum c depending on weights and the proximal
mapping results z.
"""
function weighted_sum!(weights::Array{T,1},z::Array{T,4},c::Array{T,3}) where {T<:Real}
  zSize = size(z[1,:,:,:])
  c = reshape(c,zSize[1]*zSize[2]*zSize[3])
  cTemp = zeros(eltype(c),zSize[1]*zSize[2]*zSize[3])

  for i=1:length(weights)
  zTemp = view(reshape(z,Val(2)),i,:)
  cTemp[:] = cTemp[:] + weights[i]*zTemp
  end

  for i = 1:length(c)
  c[i] = cTemp[i]
  end

end


"""
This function checks if the stopping criterion for the main loop is reached.
"""
function check_stop(cOld::Array{T,3},c::Array{T,3},tol::T) where {T<:Real}
  cSize = size(c)

  c = reshape(c,cSize[1]*cSize[2]*cSize[3])
  cOld = reshape(cOld,cSize[1]*cSize[2]*cSize[3])

  if (norm(cOld-c)/norm(cOld) + 1.0*10.0^-3.0) <= tol
    return true
  else
    return false
  end

end

function residuum(A::Array{T,4},c::Array{T,3},u::Array{T,1}) where {T<:Real}
  aSize = size(A)
  cSize = size(c)
  uSize = size(u)

  A = reshape(A,aSize[1],aSize[2]*aSize[3]*aSize[4])
  c = reshape(c,cSize[1]*cSize[2]*cSize[3])

  return (norm(A*c-u)^2/norm(u))
end
