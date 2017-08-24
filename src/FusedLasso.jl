export FusedLasso
export fusedlasso


#==============================================================================#
# Private tools
#==============================================================================#

function normalize!{T<:Real}(A::Array{T,2},b::Vector{T},energy::Vector{T})
    p=Progress(size(A,1),1,"Normalize data...")

    for i=1:size(A,1)
      energy[i] = norm(vec(A[i,:]));
      for j=1:size(A,2)
        A[i,j] = A[i,j]/energy[i]
      end
      b[i] = b[i]/energy[i]
      next!(p)
    end

end


function normalize!{T}(b::Vector{T},energy::Vector{T})
    for i=1:length(energy)
      b[i] = b[i] / energy[i]
    end
end

#==============================================================================#
# User Parameter
#==============================================================================#

type FusedLassoUserParams{T}
    verbose::Bool
    cached::Bool
    maxIter::Int64
    nhood::Array{Int64,2}
    omega::Vector{T}
    gamma::T
    lambda::T
    alpha::T
    beta::T
    lib::AbstractString
    recompile::Bool
end

function FusedLassoUserParams(T::Type;
                                # Flag for info prints
                                verbose = false,
                                # Flag for using the cached version of fused lasso
                                cached = false,
                                # Number of performed iterations
                                iterations = Int64(50),
                                # Number of proximal mapping directions
                                directions = 3,
                                # Weight for update steps
                                kappa = T(0.5),
                                # Weight for proximal mapping
                                lambdaTV = T(0.0002),
                                # Weight for soft thresholding
                                lambdaL1 = T(0.0016),
                                # Weight for gradient descent step, proximal mapping and soft thresholding
                                gamma = T(10.0^-3),
                                # Library for gradient calculation
                                lib = "",
                                # Flag to recompile selected gradient library
                                recompile = false,
                                # Others
                                kargs...
                            )

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
            warn("No corresponding neighbourhood for direction number. 3 directions will be used!")
            nhood = Int64[1 0 0;0 1 0; 0 0 1]
            omega = T[1,1,1]

	end

    return FusedLassoUserParams(verbose,cached,iterations,nhood,omega,gamma,kappa,T(lambdaTV),T(lambdaL1),lib,recompile)
end

#==============================================================================#
# Linear Problem
#==============================================================================#
type FusedLassoProblem{T}
    # System matrix
    A::Array{T,2}
    # Solution
    x::Vector{T}
    # Measurement vector
    b::Vector{T}
    # 3D dimensions of solution
    shape::Array{Int64,1}
end

type FusedLassoProblemCached{T}
    # System matrix
    A::Array{T,2}
    # Precalculation for gradient descent step
    ATA::Array{T,2}
    # Precalculation for gradient descent step
    ATb::Vector{T}
    # Solution vector
    x::Vector{T}
    # Measurement vector
    b::Vector{T}
    # 3D dimensions of solution
    shape::Array{Int64,1}
end
#==============================================================================#
# Temp Parameter
#==============================================================================#

type FusedLassoTempParams
    # Row energies of system matrix
    rowEnergy
    # Temporary vector
    y
end

#==============================================================================#
# Linear Solver
#==============================================================================#

type FusedLasso{T} <: AbstractLinearSolver
    # Could be linear problem of cached or non cached fused lasso
    linearProblem::T
    userParams::FusedLassoUserParams
    tempParams::FusedLassoTempParams
end

function FusedLasso{T}(S::Matrix{T}; shape::Array{Int64,1}=[size(S,2),1,1], kwargs...)
    userParams = FusedLassoUserParams(T;kwargs...)

    # Create linear problem for cached or non cached fused lasso
    if userParams.cached == false
      linearProblem = FusedLassoProblem(copy(S),zeros(eltype(S),shape[1]*shape[2]*shape[3]),zeros(eltype(S), div(length(S),shape[1]*shape[2]*shape[3])),shape)
    else
      linearProblem = FusedLassoProblemCached(copy(S),
                                              zeros(eltype(S),shape[1]*shape[2]*shape[3],shape[1]*shape[2]*shape[3]),
                                              zeros(eltype(S),shape[1]*shape[2]*shape[3]),
                                              zeros(eltype(S),shape[1]*shape[2]*shape[3]),
                                              zeros(eltype(S), div(length(S),shape[1]*shape[2]*shape[3])),
                                              shape
                                             )
    end

    # Call constructor interface (function below) and get linear solver object
    solver = FusedLasso(linearProblem,userParams)

    return solver
end


function FusedLasso(linearProblem,userParams::FusedLassoUserParams)

    # Create object for temporary parameters
    tempParams = FusedLassoTempParams(
                     zeros(eltype(linearProblem.A),size(linearProblem.A,1)),
                     zeros(eltype(linearProblem.x),size(linearProblem.A,linearProblem.shape[1],linearProblem.shape[2],linearProblem.shape[3]))
                     )

    # Create linear solver object
    solver = FusedLasso(
                 linearProblem,
                 userParams,
                 tempParams
                 )

    # Create shorthand notations for parameters
    S = solver.linearProblem.A
    shape = solver.linearProblem.shape
    u = solver.linearProblem.b
    energy = solver.tempParams.rowEnergy

    # Store row energies and normalize rows of system matrix
    normalize!(S,u,energy)

    #linearSolver.linearProblem.x = zeros(eltype(linearSolver.linearProblem.x),shape[1]*shape[2]*shape[3])
    solver.linearProblem.A = S

    if solver.userParams.cached == true
        println("Calculating ATA")
        #=
	BLAS.gemm!('T','N',
                   linearSolver.userParams.gamma,
                   linearSolver.linearProblem.A,
                   linearSolver.linearProblem.A,
                   zero(eltype(linearSolver.linearProblem.A)),
                   linearSolver.linearProblem.ATA)
        =#
        BLAS.syrk!('U','T',solver.userParams.gamma,
                    solver.linearProblem.A,
                    zero(eltype(solver.linearProblem.A)),
                    solver.linearProblem.ATA)
    end

    return solver
end


#==============================================================================#
# Parameter Settings
#==============================================================================#

function setParameter!(linearSolver::FusedLasso,parameter::AbstractString,data)
    if parameter == "iterations"
    elseif parameter == "directions"
    elseif parameter == "omega"
    elseif parameter == "gamma"
    elseif parameter == "lambd"
    elseif parameter == "alpha"
    elseif parameter == "beta"
    else
        warn("Invalid parameter: ",parameter)
    end
end

function setMeasurement!(linearSolver::FusedLasso,measurement::Vector)
   m = copy(measurement)
   # Multiply measurement vector with row energies of system matrix
   normalize!(m,linearSolver.tempParams.rowEnergy)
   # Store result
   linearSolver.linearProblem.b = m
end

#==============================================================================#
# Solve interface
#==============================================================================#

function solve(linearSolver::FusedLasso, measurement::Vector)
    setMeasurement!(linearSolver,measurement)
    solve(linearSolver.linearProblem,linearSolver.userParams,linearSolver.tempParams)
end

function solve(linearProblem::FusedLassoProblem,userParams::FusedLassoUserParams,tempParams::FusedLassoTempParams)
      S = linearProblem.A
      shape = linearProblem.shape
      linearProblem.x[:] = 0.0
      #linearProblem.x = zeros(eltype(linearProblem.x),shape[1]*shape[2]*shape[3])
      linearProblem.x = reshape(fusedlasso(
                reshape(S,size(S,1),shape[1],shape[2],shape[3]),
                linearProblem.b,
                reshape(linearProblem.x,shape[1],shape[2],shape[3]);
                maxIter=userParams.maxIter,
                nhood=userParams.nhood,
                omega=userParams.omega,
                lambda=userParams.lambda,
                alpha=userParams.alpha,
                beta=userParams.beta,
                gamma=userParams.gamma,
                verbose=userParams.verbose),
                shape[1]*shape[2]*shape[3]
                )
end

function solve(linearProblem::FusedLassoProblemCached,userParams::FusedLassoUserParams,tempParams::FusedLassoTempParams)
      S = linearProblem.A
      shape = linearProblem.shape
      linearProblem.x = reshape(fusedlassoCached(
                reshape(S,size(S,1),shape[1],shape[2],shape[3]),
                linearProblem.ATA,
                linearProblem.b,
                linearProblem.ATb,
                reshape(linearProblem.x,shape[1],shape[2],shape[3]);
                maxIter=userParams.maxIter,
                nhood=userParams.nhood,
                omega=userParams.omega,
                lambda=userParams.lambda,
                alpha=userParams.alpha,
                beta=userParams.beta,
                gamma=userParams.gamma,
                verbose=userParams.verbose),shape[1]*shape[2]*shape[3])
end

#==============================================================================#
# Algorithm
#==============================================================================#

type StartRange <: AbstractArray{CartesianRange,3}
  x::CartesianRange
  y::CartesianRange
  z::CartesianRange
end

Base.size(R::StartRange) = (Int(3),)
Base.IndexStyle(::Type{StartRange}) = IndexLinear()
Base.getindex(R::StartRange,i::Int) = if i==1 R.x elseif i==2 R.y elseif i==3 R.z end;

@doc """This function implements the base version of fused lasso reconstruction.

### Keyword/Optional Arguments
* 'maxIter::Int64' Maximum number of Iterations.
* 'tol::Float32' Tolerance for the stopping criterion.
* 'nhood::Array{Int64,N,3}' Neighborhood relationships for tv problem.
* 'omega::Array{Float32,N}' Vector with weights for the tv results.
* 'lambda::Float32' Scale for the update step.
* 'alpha::Float32' Weight of the tv term.
* 'beta::Float32' Weight of the l1 term.
* 'gamma::Float32' Weight for the gradient descent step.
* 'verbose' Flag for extended information output
""" ->
function fusedlasso{T}(S::Array{T,4},u::Vector{T},c::Array{T,3};
  maxIter = Int64(50),
  tol = T(5.0*10.0^-6),
  nhood = Int64[1 0 0;0 1 0; 0 0 1],
  omega = T[1 1 1],
  lambda = T(1),
  alpha = T(0.000017),
  beta = T(0.001),
  gamma = T(10.0^-3),
  verbose = false,
  kargs...
  )
  #=
  println(maxIter)
  println(tol)
  println(lambda)
  println(alpha)
  println(beta)
  println(gamma)
  =#
  gradientFunc! = gradient_base!

  cSize = size(c)
  N = size(collect(omega),1)
  z = zeros(eltype(c),N+1,cSize[1],cSize[2],cSize[3])

  y = zeros(eltype(c),cSize[1],cSize[2],cSize[3])

  t = ones(T,N+1)/(N+1)
  yTemp = Array{eltype(y)}(size(y))
  #cOld = copy(c)
  zTemp = Array{eltype(yTemp)}(size(yTemp))

  if verbose == true
    res = residuum(S,c,u)
    println("Residuum pre reconstruction: ", res)
  end
  p = Progress(Int64(maxIter)*5, 1, "Fused lasso reconstruction...")
  for i=1:maxIter

      # Calculate gradient
      gradientFunc!(c,S,u,y,gamma)
      next!(p)

      # Proximal mapping
      proxmap!(y,yTemp,z,c,N,nhood,alpha,beta,gamma,omega,lambda,t)
      next!(p)

      # Pointwise maximum
      pointwise_max!(c,z,N+1,y,yTemp)
      next!(p)

      # Update step
      update!(z,N+1,yTemp,c,lambda)
      next!(p)

      # Averaging over single results
      c = reshape(c,cSize[1],cSize[2],cSize[3])
      weighted_sum!(t,z,c)
      next!(p)
      #println("")
      #res = residuum(S,c,u)
      #println(res)
      # Evaluate stopping criterion
      #check_stop(cOld,c,tol) && break;
      #cOld = copy(c)

  end
  if verbose == true
    res = residuum(S,c,u)
    println("Residuum post reconstruction: ", res)
  end
  return c

end

function fusedlassoCached{T}(S::Array{T,4},STS::Array{T,2},u::Vector{T},STu::Vector{T},c::Array{T,3};
  maxIter = Int64(50),
  tol = T(5.0*10.0^-6),
  nhood = Int64[1 0 0;0 1 0; 0 0 1],
  omega = T[1 1 1],
  lambda = T(1),
  alpha = T(0.000017),
  beta = T(0.001),
  gamma = T(10.0^-3),
  verbose = false,
  kargs...
  )
  #=
  println(maxIter)
  println(tol)
  println(lambda)
  println(alpha)
  println(beta)
  println(gamma)
  =#
  gradientFuncCached! = gradient_base_cached!

  cSize = size(c)
  N = size(collect(omega),1)
  z = zeros(eltype(c),N+1,cSize[1],cSize[2],cSize[3])
  y = zeros(eltype(c),cSize[1],cSize[2],cSize[3])
  t = ones(T,N+1)/(N+1)
  yTemp = Array{eltype(y)}(size(y))
  #cOld = copy(c)
  zTemp = Array{eltype(yTemp)}(size(yTemp))

  if verbose == true
    res = residuum(S,c,u)
    println("Residuum pre reconstruction: ", res)
  end
  p = Progress(Int64(maxIter)*5, 1, "Fused lasso reconstruction...")
  #residuum(S,c,u)
  for i=1:maxIter

      # Calculate gradient
      gradientFuncCached!(c,STS,STu,y)
      #gradient!(c,S,u,y,gamma)
      next!(p)

      # Proximal mapping
      proxmap!(y,yTemp,z,c,N,nhood,alpha,beta,gamma,omega,lambda,t)
      next!(p)

      # Pointwise maximum
      pointwise_max!(c,z,N+1,y,yTemp)
      next!(p)

      # Update step
      update!(z,N+1,yTemp,c,lambda)
      next!(p)

      # Averaging over single results
      c = reshape(c,cSize[1],cSize[2],cSize[3])
      weighted_sum!(t,z,c)
      next!(p)
      #res = residuum(S,c,u)
      #println(res)
      # Evaluate stopping criterion
      #check_stop(cOld,c,tol) && break;
      #cOld = copy(c)

  end
  if verbose == true
    res = residuum(S,c,u)
    println("Residuum post reconstruction: ", res)
  end
  return c

end


@doc "This function calculates the error gradient according to y = γA*(Ac-u)." ->
function gradient_base!{T}(c::Array{T,3},A::Array{T,4},u::Array{T,1},y::Array{T,3},α::T)

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

function gradient_base_cached!{T<:Real}(c::Array{T,3},ATA::Array{T,2},ATu::Array{T,1},y::Array{T,3})

  aSize = size(ATA)
  cSize = size(c)
  ySize = size(y)

  c = reshape(c,cSize[1]*cSize[2]*cSize[3])
  y = reshape(y,ySize[1]*ySize[2]*ySize[3])

  BLAS.blascopy!(length(y),ATu,1,y,1)

  BLAS.symv!('U',one(T),ATA,c,one(T),y)

end

@doc "This function performs the proximal mapping." ->
function proxmap!{T<:Real}(y::Array{T,3},yTemp::Array{T,3},z::Array{T,4},c::Array{T,3},N,nhood,alpha,beta,gamma,omega,lambda,t)
  cSize = size(c)
  ySize = size(y)
  zSize = size(z)

  updateTemp = Array{eltype(y)}(ySize[1]*ySize[2]*ySize[3])

  c=reshape(c,cSize[1]*cSize[2]*cSize[3])
  y = reshape(y,ySize[1]*ySize[2]*ySize[3])
  yTemp = reshape(yTemp,ySize[1]*ySize[2]*ySize[3])

  for s=1:N

      yTemp = 2*c-reshape(z[s,:,:,:],ySize[1]*ySize[2]*ySize[3])-y;

      # Solve TV problems
      yTemp = reshape(yTemp,ySize[1],ySize[2],ySize[3])
      tv_denoise_3d_condat!(yTemp,nhood[s,:],gamma*omega[s]*alpha/t[s])

      # Soft thresholding
      softthresh!(yTemp,gamma * beta/(N*t[s]))

      # Update step
      c=reshape(c,cSize[1],cSize[2],cSize[3])
      update!(z,s,yTemp,c,lambda)
      c=reshape(c,cSize[1]*cSize[2]*cSize[3])

  end
end

@doc "This function returns the one dimensional tv problem depending on a start pixel neighbor and a direction increment." ->
function tv_get_onedim_data!{T<:Real}(tvData::Array{T,3},tvOneDim::Array{T,1},neighbor,increment,arrayCount::Array{Int64,1})

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


@doc "This function sorts the 1d tv result back into the 3d data." ->
function tv_push_onedim_data!{T<:Real}(tvData::Array{T,3},tvOneDim::Array{T,1},arrayCount::Int64,neighbor,increment)

  for i=1:arrayCount
  @inbounds tvData[neighbor] = tvOneDim[i]
  neighbor = neighbor + increment
  end

end

@doc "This function extracts 1d problems from the 3d data and starts the 1d tv function." ->
function tv_denoise_3d_condat!{T<:Real}(tvData::Array{T,3},nhood::Array{Int64,1},lambda)

  tvSize = size(tvData)
  cartRange = get_startrange(tvSize,nhood[:])
  increment = CartesianIndex((nhood[1],nhood[2],nhood[3]));
  tvOneDim = Array{eltype(tvData)}(Int64(ceil(sqrt(tvSize[1]*tvSize[2]*tvSize[3]))))
  arrayCount = Array{Int64}(1)
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


@doc "This function performs the 1d tv algorithm."
function tv_denoise_1d_condat!{T<:Real}(c::Array{T,1},width::Int64,lambda::T)

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


@doc "This function applies soft thresholding on given data y." ->
function softthresh!{T<:Real}(y::Array{T,3},threshold::T)

  ySize = size(y)
  y = reshape(y,ySize[1]*ySize[2]*ySize[3])

  y[:] = [sign(i)*max(i-threshold,0) for i in y]

end


@doc "This function performs the update step." ->
function update!{T<:Real}(z::Array{T,4},s::Int64,y::Array{T,3},c::Array{T,3},lambda::T)
  dataSize = size(c)
  flatSize = dataSize[1]*dataSize[2]*dataSize[3];
  y = reshape(y,flatSize)
  c = reshape(c,flatSize)
  yTemp = Array{T}(flatSize)
  zTemp = view(reshape(z,Val{2}),s,:)

  broadcast!(-,yTemp,y,c)

  #yTemp[:] = y[:] - c[:]
  BLAS.scale!(lambda,yTemp)
  #zTemp[:] = zTemp[:] + yTemp[:]
  broadcast!(+,zTemp,zTemp,yTemp)
  #z[s,:,:,:] = reshape(yTemp,dataSize)[:,:,:]

end


@doc "This function checks if the cartisian index exceeds size." ->
function inrange(size::Tuple{Int64,Int64,Int64},range::CartesianIndex{3})

    if range.I[1] > size[1] || range.I[1] < 1 || range.I[2] > size[2] || range.I[2] < 1 || range.I[3] > size[3] || range.I[3] < 1
      return false
    else
      return true
    end

end


@doc "This function returns a StartRange variable, which contains the start planes for the 1d tv extraction." ->
function get_startrange{T<:Real}(size::Tuple{Int64,Int64,Int64},step::Array{T,1})

  output=StartRange(
  CartesianRange(CartesianIndex((0,0,0))),
  CartesianRange(CartesianIndex((0,0,0))),
  CartesianRange(CartesianIndex((0,0,0)))
  )


  output.x = CartesianRange(CartesianIndex((1,1,1)),CartesianIndex((step[1],size[2],size[3])))
  output.y = CartesianRange(CartesianIndex((1,1,1)),CartesianIndex((size[1],step[2],size[3])))
  output.z = CartesianRange(CartesianIndex((1,1,1)),CartesianIndex((size[1],size[2],step[3])))


  if step[1] < 0
  output.x = CartesianRange(CartesianIndex((size[1]+step[1],1,1)),CartesianIndex((size[1],size[2],size[3])))
  end

  if step[2] < 0
  output.y = CartesianRange(CartesianIndex((1,size[2]+step[2],1)),CartesianIndex((size[1],size[2],size[3])))
  end

  if step[3] < 0
  output.z = CartesianRange(CartesianIndex((1,1,size[3]+step[3])),CartesianIndex((size[1],size[2],size[3])))
  end

  return output

end


@doc "This function applies the pointwise maximum step." ->
function pointwise_max!{T<:Real}(c::Array{T,3},z::Array{T,4},s::Int64,y::Array{T,3},out::Array{T,3})
  ySize = size(y)
  cSize = size(c)

  y = reshape(y,ySize[1]*ySize[2]*ySize[3])
  c = reshape(c,cSize[1]*cSize[2]*cSize[3])
  zTemp = view(reshape(z,Val{2}),s,:)
  out = reshape(out,cSize[1]*cSize[2]*cSize[3])

  yTemp = 2*c-zTemp-y;

  for j=1:length(yTemp)
    out[j] = max(yTemp[j],0)
  end

end


@doc "This function calculates the weighted sum c depending on weights and the proximal mapping results z." ->
function weighted_sum!{T<:Real}(weights::Array{T,1},z::Array{T,4},c::Array{T,3})
  zSize = size(z[1,:,:,:])
  c = reshape(c,zSize[1]*zSize[2]*zSize[3])
  cTemp = zeros(eltype(c),zSize[1]*zSize[2]*zSize[3])

  for i=1:length(weights)
  zTemp = view(reshape(z,Val{2}),i,:)
  cTemp[:] = cTemp[:] + weights[i]*zTemp
  end

  for i = 1:length(c)
  c[i] = cTemp[i]
  end

end


@doc "This function checks if the stopping criterion for the main loop is reached." ->
function check_stop{T<:Real}(cOld::Array{T,3},c::Array{T,3},tol::T)
  cSize = size(c)

  c = reshape(c,cSize[1]*cSize[2]*cSize[3])
  cOld = reshape(cOld,cSize[1]*cSize[2]*cSize[3])

  #println(norm(cOld-c)/norm(cOld)+1.0*10^-3.0)

  if (norm(cOld-c)/norm(cOld) + 1.0*10.0^-3.0) <= tol
    return true
  else
    return false
  end

end

function residuum{T<:Real}(A::Array{T,4},c::Array{T,3},u::Array{T,1})
aSize = size(A)
cSize = size(c)
uSize = size(u)

A = reshape(A,aSize[1],aSize[2]*aSize[3]*aSize[4])
c = reshape(c,cSize[1]*cSize[2]*cSize[3])

#println("Residuum norm: ", (norm(A*c-u)^2)/norm(u))

#println("Norm u: ", norm(u))
#return((norm(A*c-u)^2)/norm(u))
return(norm(A*c-u)^2/norm(u))

end
