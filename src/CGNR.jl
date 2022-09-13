export cgnr

mutable struct CGNR{vecT,T,Tsparse} <: AbstractLinearSolver
  S
  SHWS
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
* (weights::vecT=eltype(S)[]) - weights for the data term
* (sparseTrafo=nothing)             - sparsifying transform
* (enforceReal::Bool=false)         - constrain the solution to be real
* (enforcePositive::Bool=false)     - constrain the solution to have positive real part
* (iterations::Int64=10)            - number of iterations
* (`relTol::Float64=eps()`)         - rel tolerance for stopping criterion
"""
function CGNR(S, x::vecT=zeros(eltype(S),size(S,2)); λ::Real=0.0, reg::R = Regularization("L2", λ)
              , weights::vecT=similar(x,0)
              , sparseTrafo=nothing
              , enforceReal::Bool=false
              , enforcePositive::Bool=false
              , iterations::Int64=10
              , relTol::Float64=eps()
              , normalizeReg::Bool=false
              , kargs...) where {vecT<:AbstractVector,R<:Union{Regularization, Vector{Regularization}}}

  if typeof(reg)==Vector{Regularization}
    reg = reg[1]
  end

  if (reg.prox!) != (proxL2!)
    @error "CGNR only supports L2 regularizer"
  end

  M,N = size(S)
  T = eltype(S)
  cl = similar(x,N)
  rl = similar(x,M)     #residual vector
  zl = similar(x,N)     #temporary vector
  pl = similar(x,N)     #temporary vector
  vl = similar(x,N)     #temporary vector
  xl = similar(x,M)     #temporary vector
  αl = zero(T)        #temporary scalar
  βl = zero(T)        #temporary scalar
  ζl = zero(T)        #temporary scalar
  SHWS = normalOperator(S, isempty(weights) ? opEye(T, size(S,1)) : WeightingOp(weights))

  return CGNR(S,SHWS,
             reg,cl,rl,zl,pl,vl,xl,αl,βl,ζl,
             weights,enforceReal,enforcePositive,sparseTrafo,iterations,relTol,0.0,normalizeReg,1.0)
end

"""
init!(solver::CGNR{vecT,T,Tsparse}, u::vecT
              ; S::matT=solver.S
              , cl::vecT=similar(u,0)
              , weights::vecT=solver.weights) where {vecT,T,Tsparse,matT}

(re-) initializes the CGNR iterator
"""
function init!(solver::CGNR{vecT,T,Tsparse}, u::vecT
              ; S::matT=solver.S
              , cl::vecT=similar(u,0)
              , weights::vecT=solver.weights) where {vecT,T,Tsparse,matT}

  solver.S = S
  # TODO, the following line is called a second time...
  #solver.SHWS = normalOperator(S, isempty(weights) ? opEye(T, size(S,1)) : WeightingOp(weights))
  if isempty(cl)
    solver.cl[:] .= zero(T)
  else
    solver.cl[:] .= cl
  end
  solver.rl[:] .= u - S*solver.cl
  solver.zl[:] .= zero(T)     #temporary vector
  solver.pl[:] .= zero(T)     #temporary vector
  solver.vl[:] .= zero(T)     #temporary vector
  solver.xl[:] .= zero(T)     #temporary vector
  solver.αl = zero(T)        #temporary scalar
  solver.βl = zero(T)        #temporary scalar
  solver.ζl = zero(T)        #temporary scalar

  #zl = Sᶜ*rl, where ᶜ denotes complex conjugation
  if !isempty(weights)
    solver.xl[:] .= solver.rl .* weights
    ## gemv!('C',one(T), S, xl, zero(T), zl)
    mul!(solver.zl, adjoint(S), solver.xl)
  else
    ## gemv!('C',one(T), S, rl, zero(T), zl)
    mul!(solver.zl, adjoint(S), solver.rl)
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

solves Thikhonov-regularized inverse problem using CGNR.

# Arguments
* `solver::CGNR                         - the solver containing both system matrix and regularizer
* `u::vecT`                             - data vector
* (`S::matT=solver.S`)                  - operator for the data-term of the problem
* (`startVector::vecT=similar(u,0)`)    - initial guess for the solution
* (`weights::Vector{T}=solver.weights`) - weights for the data term
* (`solverInfo=nothing`)                - solverInfo for logging

when a `SolverInfo` objects is passed, the residuals `solver.zl` are stored in `solverInfo.convMeas`.
"""
function solve(solver::CGNR{vecT,T,Tsparse}, u::vecT; S::matT=solver.S, startVector::vecT=similar(u,0), weights::vecT=solver.weights, solverInfo=nothing, kargs...) where {vecT,T,Tsparse,matT}
  # initialize solver parameters
  init!(solver, u; S=S, cl=startVector, weights=weights)

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
function iterate(solver::CGNR{vecT,T,Tsparse}, iteration::Int=0) where {vecT,T,Tsparse}
    if done(solver,iteration)
      applyConstraints(solver.cl, solver.sparseTrafo, solver.enforceReal, solver.enforcePositive)
      return nothing
    end

    mul!(solver.vl, solver.SHWS, solver.pl)

    # αl = zlᴴ⋅zl/(vlᴴ⋅vl+λ*plᴴ⋅pl)
    solver.ζl= norm(solver.zl)^2
    normvl = dot(solver.pl,solver.vl) 

    if solver.reg.λ > 0
      solver.αl = solver.ζl/(normvl+solver.regFac*solver.reg.λ*norm(solver.pl)^2)
    else
      solver.αl = solver.ζl/normvl
    end

    #cl += αl*pl
    BLAS.axpy!(solver.αl,solver.pl,solver.cl)

    #rl += -αl*vl
    BLAS.axpy!(-solver.αl,solver.vl,solver.zl)
    
    if solver.reg.λ > 0
      BLAS.axpy!(-solver.reg.λ*solver.αl,solver.pl,solver.zl)
    end

    # βl = zl₊₁ᴴ⋅zl₊₁/zlᴴ⋅zl
    solver.βl = dot(solver.zl,solver.zl)/solver.ζl

    #pl = zl + βl*pl
    rmul!(solver.pl,solver.βl)
    BLAS.axpy!(one(eltype(solver.S)),solver.zl,solver.pl)
    return solver.zl, iteration+1
end


#=
function iterate(solver::CGNR{T,Tsparse}, iteration::Int=0) where {T,Tsparse}
    if done(solver,iteration)
      applyConstraints(solver.cl, solver.sparseTrafo, solver.enforceReal, solver.enforcePositive)
      return nothing
    end

    #vl = Sᵗ*pl
    ##gemv!('N',one(T), S, pl, zero(T), vl)
    solver.vl[:] .= solver.S*solver.pl

    # αl = zlᴴ⋅zl/(vlᴴ⋅vl+λ*plᴴ⋅pl)
    solver.ζl= norm(solver.zl)^2
    normvl = isempty(solver.weights) ? dot(solver.vl,solver.vl) : dot(solver.vl,solver.weights.*solver.vl)

    if solver.reg.λ > 0
      solver.αl = solver.ζl/(normvl+solver.reg.λ*norm(solver.pl)^2)
    else
      solver.αl = solver.ζl/normvl
    end

    #cl += αl*pl
    BLAS.axpy!(solver.αl,solver.pl,solver.cl)

    #rl += -αl*vl
    BLAS.axpy!(-solver.αl,solver.vl,solver.rl)

    #zl = Sᶜ*rl-λ*cl
    if !isempty(solver.weights)
      solver.xl[:] .= solver.rl .* solver.weights
      ##gemv!('C',one(T), S, xl, zero(T), zl)
      solver.zl[:] .= adjoint(solver.S)*solver.xl
    else
      ##gemv!('C',one(T), S, rl, zero(T), zl)
      solver.zl[:] .= adjoint(solver.S)*solver.rl
    end
    if solver.reg.λ > 0
      BLAS.axpy!(-solver.reg.λ,solver.cl,solver.zl)
    end

    # βl = zl₊₁ᴴ⋅zl₊₁/zlᴴ⋅zl
    solver.βl = dot(solver.zl,solver.zl)/solver.ζl

    #pl = zl + βl*pl
    rmul!(solver.pl,solver.βl)
    BLAS.axpy!(one(eltype(solver.S)),solver.zl,solver.pl)
    return solver.rl, iteration+1
end
=#

function converged(solver::CGNR)
  return norm(solver.zl)/solver.z0 <= solver.relTol
end

@inline done(solver::CGNR,iteration::Int) = converged(solver) || iteration>=min(solver.iterations, size(solver.S,2))
