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
This function applies soft thresholding on given data y.
"""
function softthresh!(y::Array{T,3}, threshold) where {T<:Real}

  ySize = size(y)
  y = reshape(y,ySize[1]*ySize[2]*ySize[3])

  y[:] = [sign(i)*max(i-threshold,0) for i in y]
  return
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
  return
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
  return
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
  return
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
