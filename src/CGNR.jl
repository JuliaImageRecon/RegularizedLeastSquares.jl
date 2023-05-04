export cgnr, CGNR

mutable struct CGNR{matT,opT,vecT,T,Tsparse} <: AbstractLinearSolver
  A::matT
  AᴴA::opT
  reg::Regularization
  cl::vecT
  rl::vecT
  zl::vecT
  pl::vecT
  vl::vecT
  xl::vecT
  αl::T
  βl::T
  ζl::T
  weights::vecT
  enforceReal::Bool
  enforcePositive::Bool
  sparseTrafo::Tsparse
  iterations::Int64
  relTol::Float64
  z0::Float64
  normalizeReg::Bool
  regFac::Float64
end

"""
    CGNR(A, x::vecT; λ = 0.0, reg = Regularization("L2", λ), kargs...) where vecT

creates an `CGNR` object for the system matrix `A`.

# Arguments
* `A`                               - system matrix
* `x::vecT`                         - Array with the same type and size as the solution
* (`λ=0.0`)                         - Regularization paramter
* (`reg=Regularization("L2", λ)`)   - Regularization object
* (weights::vecT=eltype(A)[]) - weights for the data term
* (sparseTrafo=nothing)             - sparsifying transform
* (enforceReal::Bool=false)         - constrain the solution to be real
* (enforcePositive::Bool=false)     - constrain the solution to have positive real part
* (iterations::Int64=10)            - number of iterations
* (`relTol::Float64=eps()`)         - rel tolerance for stopping criterion
"""
function CGNR(A, x::vecT=zeros(eltype(A),size(A,2)); λ::Real=0.0, reg::R = Regularization("L2", λ)
              , weights::vecT=similar(x,0)
              , AᴴA::opT=nothing
              , sparseTrafo=nothing
              , enforceReal::Bool=false
              , enforcePositive::Bool=false
              , iterations::Int64=10
              , relTol::Float64=eps()
              , normalizeReg::Bool=false
              , kargs...) where {opT,vecT<:AbstractVector,R<:Union{Regularization, Vector{Regularization}}}

  if typeof(reg)==Vector{Regularization}
    reg = reg[1]
  end

  if (reg.prox!) != (proxL2!)
    @error "CGNR only supports L2 regularizer"
  end

  if AᴴA == nothing
    AᴴA = A'*A
  end

  M, N = size(A)
  T = eltype(A)
  cl = similar(x,N)
  rl = similar(x,M)     #residual vector
  zl = similar(x,N)     #temporary vector
  pl = similar(x,N)     #temporary vector
  vl = similar(x,N)     #temporary vector
  xl = similar(x,M)     #temporary vector
  αl = zero(T)        #temporary scalar
  βl = zero(T)        #temporary scalar
  ζl = zero(T)        #temporary scalar

  return CGNR(A, AᴴA,
             reg,cl,rl,zl,pl,vl,xl,αl,βl,ζl,
             weights,enforceReal,enforcePositive,sparseTrafo,iterations,relTol,0.0,normalizeReg,1.0)
end

"""
init!(solver::CGNR{vecT,T,Tsparse}, u::vecT
              ; cl::vecT=similar(u,0)
              , weights::vecT=solver.weights) where {vecT,T,Tsparse,matT}

(re-) initializes the CGNR iterator
"""
function init!(solver::CGNR, u::vecT
              ; cl::vecT=similar(u,0)) where {vecT}
  T = eltype(solver.A)

  if isempty(cl)
    solver.cl[:] .= zero(T)
  else
    solver.cl[:] .= cl
  end
  solver.rl[:] .= u - solver.A*solver.cl
  solver.zl[:] .= zero(T)     #temporary vector
  solver.pl[:] .= zero(T)     #temporary vector
  solver.vl[:] .= zero(T)     #temporary vector
  solver.xl[:] .= zero(T)     #temporary vector
  solver.αl = zero(T)        #temporary scalar
  solver.βl = zero(T)        #temporary scalar
  solver.ζl = zero(T)        #temporary scalar

  #zl = Aᶜ*rl, where ᶜ denotes complex conjugation
  if !isempty(solver.weights)
    solver.xl[:] .= solver.rl .* solver.weights
    mul!(solver.zl, adjoint(solver.A), solver.xl)
  else
    mul!(solver.zl, adjoint(solver.A), solver.rl)
  end
  solver.z0 = norm(solver.zl)
  copyto!(solver.pl,solver.zl)

  # normalization of regularization parameters
  if solver.normalizeReg
    solver.regFac = norm(u,1)/length(u)
  else
    solver.regFac = 1.0
  end
end

"""
    solve(solver::CGNR, u::vecT) where vecT

solves Tikhonov-regularized inverse problem using CGNR.

# Arguments
* `solver::CGNR                         - the solver containing both system matrix and regularizer
* `u::vecT`                             - data vector
* (`startVector::vecT=similar(u,0)`)    - initial guess for the solution
* (`solverInfo=nothing`)                - solverInfo for logging

when a `SolverInfo` objects is passed, the residuals `solver.zl` are stored in `solverInfo.convMeas`.
"""
function solve(solver::CGNR, u::vecT;  startVector::vecT=similar(u,0), solverInfo=nothing, kargs...) where {vecT}
  # initialize solver parameters
  init!(solver, u; cl=startVector)

  # log solver information
  solverInfo != nothing && storeInfo(solverInfo,solver.cl,norm(solver.zl))

  # perform CGNR iterations
  for (iteration, item) = enumerate(solver)
    solverInfo != nothing && storeInfo(solverInfo,solver.cl,norm(solver.zl))
  end

  return solver.cl
end


"""
  iterate(solver::CGNR{vecT,T,Tsparse}, iteration::Int=0) where {vecT,T,Tsparse}

performs one CGNR iteration.
"""
function iterate(solver::CGNR, iteration::Int=0) 
    if done(solver,iteration)
      applyConstraints(solver.cl, solver.sparseTrafo, solver.enforceReal, solver.enforcePositive)
      return nothing
    end

    mul!(solver.vl, solver.AᴴA, solver.pl)

    solver.ζl= norm(solver.zl)^2
    normvl = dot(solver.pl,solver.vl) 

    if solver.reg.λ > 0
      solver.αl = solver.ζl/(normvl+solver.regFac*solver.reg.λ*norm(solver.pl)^2)
    else
      solver.αl = solver.ζl/normvl
    end

    BLAS.axpy!(solver.αl,solver.pl,solver.cl)

    BLAS.axpy!(-solver.αl,solver.vl,solver.zl)
    
    if solver.reg.λ > 0
      BLAS.axpy!(-solver.reg.λ*solver.αl,solver.pl,solver.zl)
    end

    solver.βl = dot(solver.zl,solver.zl)/solver.ζl

    rmul!(solver.pl,solver.βl)
    BLAS.axpy!(one(eltype(solver.A)),solver.zl,solver.pl)
    return solver.zl, iteration+1
end


function converged(solver::CGNR)
  return norm(solver.zl)/solver.z0 <= solver.relTol
end

@inline done(solver::CGNR,iteration::Int) = converged(solver) || iteration>=min(solver.iterations, size(solver.A,2))
