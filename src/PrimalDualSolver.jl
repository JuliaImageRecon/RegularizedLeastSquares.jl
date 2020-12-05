export primaldualsolver

mutable struct PrimalDualSolver{T,S} <: AbstractLinearSolver
    S::Matrix{T}
    reg::Vector{Regularization}
    regName::Vector{String}
    gradientOp::S
    u::Vector{T}
    c::Vector{T}
    cO::Vector{T}
    y1::Vector{T}
    y2::Vector{T}
    σ::T
    τ::T
    ϵ::T
    PrimalDualGap::T
    enforceReal::Bool
    enforcePositive::Bool
    iterations::Int64
    shape::NTuple{2,Int64}
end

function PrimalDualSolver(S::Matrix{T}; b=nothing, λ=1e-4, reg = nothing
			 , regName = "L1"
                         , gradientOp = nothing
                         , enforceReal::Bool=false
                         , enforcePositive::Bool=false
                         , iterations::Int64=10
                         , σ=1.0
                         , τ=1.0
                         , ϵ=1e-10
                         , PrimalDualGap=1.0
                         , shape::NTuple{2,Int64}=(0,0)
                         , kargs...) where T

  if reg == nothing
    if typeof(λ)<:AbstractFloat && typeof(regName)==String
        reg = [Regularization(regName, T(λ); kargs...)]
    elseif typeof(λ)<:AbstractVector && typeof(regName)<:AbstractVector{String}
        reg = Regularization(regName, T(λ); kargs...)
    else
        error("could not initialite regularization")
    end
  end

  M,N = size(S)

  if shape == (0,0)
    shape = (Int(sqrt(N)),Int(sqrt(N)))
  else
    shape = shape
  end

  if typeof(regName)==String
      regName = [regName]
  end
  
  if regName[1] == "L1"
    gradientOp = opEye(T,N) #UniformScaling(one(T))
  elseif regName[1] == "TV"
    gradientOp = gradientOperator(T,shape)
  end

  if b != nothing
    u = b
  else
    u = zeros(T,M)
  end

  c = zeros(T,N)
  cO = zeros(T,N)
  y1 = zeros(T,M)
  y2 = zeros(T,size(gradientOp*c,1))

  return PrimalDualSolver(S,reg,regName,gradientOp,u,c,cO,y1,y2,T(σ),T(τ),T(ϵ),T(PrimalDualGap),enforceReal,enforcePositive,iterations,shape)
end

function init!(solver::PrimalDualSolver; S::Matrix{T}=solver.S, u::Vector{T}=T[], c::Vector{T}=T[],
    PrimalDualGap::T=solver.PrimalDualGap) where {T,R}

  solver.u[:] .= u
  solver.PrimalDualGap = (1/2)*(norm(solver.u,2))^2

  # start vector
  if isempty(c)
    solver.c[:] .= zero(T)
  else
    solver.c[:] .= c
  end
  solver.cO[:] .= zero(T)
  solver.y1[:] .= zero(T)
  solver.y2[:] .= zero(T)

end

function solve(solver::PrimalDualSolver, u::Vector{T}; S::Matrix{T}=solver.S, startVector::Vector{T}=eltype(S)[]
              , solverInfo=nothing, PrimalDualGap::T=solver.PrimalDualGap, kargs...) where {T}

  # initialize solver parameters
  init!(solver; S=S, u=u, c=startVector, PrimalDualGap=solver.PrimalDualGap)

  # log solver information
  solverInfo != nothing && storeInfo(solverInfo,solver.c,solver.cO,solver.y1,solver.y2)

  # perform the iterations
  for item in solver
    solverInfo != nothing && storeInfo(solverInfo,solver.c,solver.cO,solver.y1,solver.y2)
  end

  return solver.c
end

function iterate(solver::PrimalDualSolver, iteration::Int=0)
  if done(solver,iteration) return nothing end

  # updating dual variables
  for i=1:length(solver.reg)
      solver.y1 .= (solver.y1 + solver.σ*(solver.S*solver.c - solver.u))./(1+solver.σ)
      if solver.regName[1] == "L1"
          solver.y2 .= ProxL1Conj(solver.y2 + solver.σ*solver.gradientOp*solver.c, solver.reg[1].λ, solver.shape)
      elseif solver.regName[1] == "TV"
          solver.y2 .= ProxTVConj(solver.y2 + solver.σ*solver.gradientOp*solver.c, solver.reg[1].λ, solver.shape)
      end
  end

  # updating primal variable
  for i=1:length(solver.reg)
      solver.c += - solver.τ*(adjoint(solver.S)*solver.y1 + adjoint(solver.gradientOp)*solver.y2)
  end

  applyConstraints(solver.c, nothing, solver.enforceReal, solver.enforcePositive)

  # updating convergence measure
  for i=1:length(solver.reg)
    solver.PrimalDualGap = abs((1/2)*norm(solver.S*solver.c-solver.u)^2 + solver.reg[1].λ*norm(solver.c,1) + (1/2)*norm(solver.y1,2)^2 + dot(solver.y1,solver.u))
  end

  return solver.y1, iteration+1

end

function converged(solver::PrimalDualSolver)
  if (solver.PrimalDualGap) >= solver.ϵ
    return false
  end

  return true
end

@inline done(solver::PrimalDualSolver,iteration::Int) = converged(solver) || iteration>=solver.iterations

# Proximal map of the convex conjugate of the l1 norm
function ProxL1Conj(x::Vector{T},α::T,shape::NTuple{2,Int64}) where T
   m,n = shape
   p1 = reshape(x,m,n)

   # threshold p1
   for j=1:n, i=1:m
       p1[i,j] = sign(p1[i,j])*min(norm(p1[i,j]),α)
   end
   for i=1:m
       p1[i,n] = sign(p1[i,n])*min(abs(p1[i,n]),α)
   end

   return vcat(vec(p1))
end

# Gradient
function BB(u::Vector{T}, shape::NTuple{2,Int64}) where T
 m,n = shape
 u = reshape(u,m,n)

 p1 = zeros(T,m,n);  p2 = zeros(T,m,n);  p3 = zeros(T,m,n);  p4 = zeros(T,m,n)
 p5 = zeros(T,m,n);  p6 = zeros(T,m,n);  p7 = zeros(T,m,n);  p8 = zeros(T,m,n)

    p1[1:m-1,1:n] = u[1:m-1,1:n]-u[2:m,1:n]
    p2[1:m,1:n-1] = u[1:m,1:n-1]-u[1:m,2:n]
    p3[1:m-1,1:n-1] = u[1:m-1,1:n-1]-u[2:m,2:n]
    p4[1:m-1,2:n] = u[1:m-1,2:n]-u[2:m,1:n-1]
    p5[1:m-2,1:n-1] = u[1:m-2,1:n-1]-u[3:m,2:n]
    p6[1:m-2,2:n] = u[1:m-2,2:n]-u[3:m,1:n-1]
    p7[1:m-1,1:n-2] = u[1:m-1,1:n-2]-u[2:m,3:n]
    p8[1:m-1,3:n] = u[1:m-1,3:n]-u[2:m,1:n-2]

 return vcat(vec(p1),vec(p2),vec(p3),vec(p4),vec(p5),vec(p6),vec(p7),vec(p8))
end

# Divergence
function BBS(u::Array{T},shape::NTuple{2,Int64}) where T
   m,n = shape
   p1 = reshape(u[1:m*n],m,n);            p2 = reshape(u[m*n+1:2*n*m],m,n)
   p3 = reshape(u[2*m*n+1:3*m*n],m,n);    p4 = reshape(u[3*m*n+1:4*m*n],m,n)
   p5 = reshape(u[4*m*n+1:5*m*n],m,n);    p6 = reshape(u[5*m*n+1:6*m*n],m,n)
   p7 = reshape(u[6*m*n+1:7*m*n],m,n);    p8 = reshape(u[7*m*n+1:end],m,n)

   x = zeros(T, m, n)

   # points in the corners
   x[1,1] = p1[1,1]+p2[1,1]+p3[1,1]+p4[1,1]+p5[1,1]+p6[1,1]+p7[1,1]+p8[1,1]
   x[m,1] = p2[m,1]-p1[m-1,1]+p3[m,1]+p4[m,1]-p4[m-1,2]+p5[m,1]+p6[m,1]-p6[m-2,2]+p7[m,1]+p8[m,1]-p8[m-1,3]
   x[1,n] = p1[1,n]-p2[1,n-1]+p3[1,n]+p4[1,n]+p5[1,n]+p6[1,n]+p7[1,n]+p8[1,n]
   x[m,n] = -p1[m-1,n]-p2[m,n-1]-p3[m-1,n-1]-p5[m-2,n-1]-p7[m-1,n-2]

   # remaining points with i=1 (first row)
   x[1,2:n-1] = p1[1,2:n-1]+p2[1,2:n-1]-p2[1,1:n-2]+p3[1,2:n-1]+p4[1,2:n-1]+p5[1,2:n-1]+p6[1,2:n-1]+p7[1,2:n-1]+p8[1,2:n-1]
   # remaining points with i=2 (second row)
   x[2,3:n-2] = p1[2,3:n-2]-p1[1,3:n-2]+p2[2,3:n-2]-p2[2,2:n-3]+p3[2,3:n-2]-p3[1,2:n-3]+p4[2,3:n-2]-p4[1,4:n-1]+p5[2,3:n-2]+p6[2,3:n-2]+p7[2,3:n-2]-p7[1,1:n-4]+p8[2,3:n-2]-p8[1,5:n]
   #remaining points with i=m (last row)
   x[m,3:n-2] = -p1[m-1,3:n-2]+p2[m,3:n-2]-p2[m,2:n-3]-p3[m-1,2:n-3]+p4[m,3:n-2]-p4[m-1,4:n-1]+p5[m,3:n-2]-p5[m-2,2:n-3]+p6[m,3:n-2]-p6[m-2,4:n-1]+p7[m,3:n-2]-p7[m-1,1:n-4]+p8[m,3:n-2]-p8[m-1,5:n]
   # remaingin points with j=1 (first column)
   x[3:m,1] = p1[3:m,1]-p1[2:m-1,1]+p2[3:m,1]+p3[3:m,1]+p4[3:m,1]-p4[2:m-1,2]+p5[3:m,1]+p6[3:m,1]-p6[1:m-2,2]+p7[3:m,1]+p8[3:m,1]-p8[2:m-1,3]
   # remaingin points with j=2 (second column)
   x[3:m,2] = p1[3:m,2]-p1[2:m-1,2]+p2[3:m,2]-p2[3:m,1]+p3[3:m,2]-p3[2:m-1,1]+p4[3:m,2]-p4[2:m-1,3]+p5[3:m,2]-p5[1:m-2,1]+p6[3:m,2]-p6[1:m-2,3]+p7[3:m,2]+p8[3:m,2]-p8[2:m-1,4]
   #remaingin points with j=n (last column)
   x[3:m,n] = p1[3:m,n]-p1[2:m-1,n]-p2[3:m,n-1]-p3[2:m-1,n-1]+p4[3:m,n]+p5[3:m,n]-p5[1:m-2,n-1]+p6[3:m,n]+p7[3:m,n]-p7[2:m-1,n-2]+p8[3:m,n]
   #remaingin points with j=n-1
   x[3:m,n-1] = p1[3:m,n-1]-p1[2:m-1,n-1]+p2[3:m,n-1]-p2[3:m,n-2]+p3[3:m,n-1]-p3[2:m-1,n-2]+p4[3:m,n-1]-p4[2:m-1,n]+p5[3:m,n-1]-p5[1:m-2,n-2]+p6[3:m,n-1]-p6[1:m-2,n]+p7[3:m,n-1]-p7[2:m-1,n-3]+p8[3:m,n-1]
   #remaingin points with j=n-2
   x[3:m,n-2] = p1[3:m,n-2]-p1[2:m-1,n-2]+p2[3:m,n-2]-p2[3:m,n-3]+p3[3:m,n-2]-p3[2:m-1,n-3]+p4[3:m,n-2]-p4[2:m-1,n-1]+p5[3:m,n-2]-p5[1:m-2,n-3]+p6[3:m,n-2]-p6[1:m-2,n-1]+p7[3:m,n-2]-p7[2:m-1,n-4]+p8[3:m,n-2]-p8[2:m-1,n]

   #all remaining (inner) matrix elements
   x[3:m-1,3:n-3] = p1[3:m-1,3:n-3]-p1[2:m-2,3:n-3]+p2[3:m-1,3:n-3]-p2[3:m-1,2:n-4]+p3[3:m-1,3:n-3]-p3[2:m-2,2:n-4]+p4[3:m-1,3:n-3]-p4[2:m-2,4:n-2]+p5[3:m-1,3:n-3]-p5[1:m-3,2:n-4]+p6[3:m-1,3:n-3]-p6[1:m-3,4:n-2]+p7[3:m-1,3:n-3]-p7[2:m-2,1:n-5]+p8[3:m-1,3:n-3]-p8[2:m-2,5:n-1]

   return vec(x)
end

function gradientOperator(::Type{T},shape::NTuple{2,Int64}) where T
   M,N = shape
   ncol = M*N
   nrow = 8*M*N
   return LinearOperator{T}(nrow,ncol,false,false,x->BB(x,shape),nothing,x->BBS(x,shape))
end

# Proximal map of the convex conjugate of the debiasing function
function ProxTVConj(x::Vector{T},α::T,shape::NTuple{2,Int64}) where T
   m,n = shape
   p1 = reshape(x[1:m*n],m,n)
   p2 = reshape(x[m*n+1:2*m*n],m,n)
   p3 = reshape(x[2*m*n+1:3*m*n],m,n)
   p4 = reshape(x[3*m*n+1:4*m*n],m,n)
   p5 = reshape(x[4*m*n+1:5*m*n],m,n)
   p6 = reshape(x[5*m*n+1:6*m*n],m,n)
   p7 = reshape(x[6*m*n+1:7*m*n],m,n)
   p8 = reshape(x[7*m*n+1:end],m,n)

   # threshold p1
   for j=1:n, i=1:m
       p1[i,j] = (sqrt(5)-2)*sign(p1[i,j])*min(norm([p1[i,j],p2[i,j],p3[i,j],p4[i,j],p5[i,j],p6[i,j],p7[i,j],p8[i,j]]),α)
   end
   for i=1:m
       p1[i,n] = (sqrt(5)-2)*sign(p1[i,n])*min(abs(p1[i,n]),α)
   end

   # threshold p2
   for j=1:n, i=1:m
       p2[i,j] = (sqrt(5)-2)*sign(p2[i,j])*min(norm([p1[i,j],p2[i,j],p3[i,j],p4[i,j],p5[i,j],p6[i,j],p7[i,j],p8[i,j]]),α)
   end
   for j=1:n
       p2[m,j] = (sqrt(5)-2)*sign(p2[m,j])*min(abs(p2[m,j]),α)
   end

   # threshold p3
   for j=1:n, i=1:m
       p3[i,j] = (sqrt(5)-3*sqrt(2)/2)*sign(p3[i,j])*min(norm([p1[i,j],p2[i,j],p3[i,j],p4[i,j],p5[i,j],p6[i,j],p7[i,j],p8[i,j]]),α)
   end
   for i=1:m
       p3[i,n] = (sqrt(5)-3*sqrt(2)/2)*sign(p3[i,n])*min(abs(p3[i,n]),α)
   end
   for j=1:n
       p3[m,j] = (sqrt(5)-3*sqrt(2)/2)*sign(p3[m,j])*min(abs(p3[m,j]),α)
   end

   # threshold p4
   for j=1:n, i=1:m
       p4[i,j] = (sqrt(5)-3*sqrt(2)/2)*sign(p4[i,j])*min(norm([p1[i,j],p2[i,j],p3[i,j],p4[i,j],p5[i,j],p6[i,j],p7[i,j],p8[i,j]]),α)
   end
   for i=1:m
       p4[i,n] = (sqrt(5)-3*sqrt(2)/2)*sign(p4[i,n])*min(abs(p4[i,n]),α)
   end
   for j=1:n
       p4[m,j] = (sqrt(5)-3*sqrt(2)/2)*sign(p4[m,j])*min(abs(p4[m,j]),α)
   end

   # threshold p5
   for j=1:n, i=1:m
       p5[i,j] = ((1+sqrt(2)-sqrt(5))/2)*sign(p5[i,j])*min(norm([p1[i,j],p2[i,j],p3[i,j],p4[i,j],p5[i,j],p6[i,j],p7[i,j],p8[i,j]]),α)
   end
   for i=1:m
       p5[i,n] = ((1+sqrt(2)-sqrt(5))/2)*sign(p5[i,n])*min(abs(p5[i,n]),α)
   end
   for j=1:n
       p5[m,j] = ((1+sqrt(2)-sqrt(5))/2)*sign(p5[m,j])*min(abs(p5[m,j]),α)
   end

   # threshold p6
   for j=1:n, i=1:m
       p6[i,j] = ((1+sqrt(2)-sqrt(5))/2)*sign(p6[i,j])*min(norm([p1[i,j],p2[i,j],p3[i,j],p4[i,j],p5[i,j],p6[i,j],p7[i,j],p8[i,j]]),α)
   end
   for i=1:m
       p6[i,n] = ((1+sqrt(2)-sqrt(5))/2)*sign(p6[i,n])*min(abs(p6[i,n]),α)
   end
   for j=1:n
       p6[m,j] = ((1+sqrt(2)-sqrt(5))/2)*sign(p6[m,j])*min(abs(p6[m,j]),α)
   end

   # threshold p7
   for j=1:n, i=1:m
       p7[i,j] = ((1+sqrt(2)-sqrt(5))/2)*sign(p7[i,j])*min(norm([p1[i,j],p2[i,j],p3[i,j],p4[i,j],p5[i,j],p6[i,j],p7[i,j],p8[i,j]]),α)
   end
   for i=1:m
       p7[i,n] = ((1+sqrt(2)-sqrt(5))/2)*sign(p7[i,n])*min(abs(p7[i,n]),α)
   end
   for j=1:n
       p7[m,j] = ((1+sqrt(2)-sqrt(5))/2)*sign(p7[m,j])*min(abs(p7[m,j]),α)
   end

   # threshold p8
   for j=1:n, i=1:m
       p8[i,j] = ((1+sqrt(2)-sqrt(5))/2)*sign(p8[i,j])*min(norm([p1[i,j],p2[i,j],p3[i,j],p4[i,j],p5[i,j],p6[i,j],p7[i,j],p8[i,j]]),α)
   end
   for i=1:m
       p8[i,n] = ((1+sqrt(2)-sqrt(5))/2)*sign(p8[i,n])*min(abs(p8[i,n]),α)
   end
   for j=1:n
       p8[m,j] = ((1+sqrt(2)-sqrt(5))/2)*sign(p8[m,j])*min(abs(p8[m,j]),α)
   end

   return vcat(vec(p1),vec(p2),vec(p3),vec(p4),vec(p5),vec(p6),vec(p7),vec(p8))
end
