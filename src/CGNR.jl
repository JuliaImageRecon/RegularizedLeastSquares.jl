export cgnr

mutable struct CGNR{T,Tsparse} <: AbstractLinearSolver
  S
  reg::Regularization
  cl::Vector{T}
  rl::Vector{T}
  zl::Vector{T}
  pl::Vector{T}
  vl::Vector{T}
  xl::Vector{T}
  αl::T
  βl::T
  ζl::T
  weights::Vector{T}
  enforceReal::Bool
  enforcePositive::Bool
  sparseTrafo::Tsparse
  iterations::Int64
end

"""
    CGNR(A; λ = 0.0, reg = Regularization("L2", λ), kargs...)

creates an `CGNR` object for the system matrix `A`.

# Arguments
* `A`                               - system matrix
* (`λ=0.0`)                         - Regularization paramter
* (`reg=Regularization("L2", λ)`)   - Regularization object
* (weights::Vector{WT}=eltype(S)[]) - weights for the data term
* (sparseTrafo=nothing)             - sparsifying transform
* (enforceReal::Bool=false)         - constrain the solution to be real
* (enforcePositive::Bool=false)     - constrain the solution to have positive real part
* (iterations::Int64=10)            - number of iterations
"""
function CGNR(S; λ::Real=0.0, reg = Regularization("L2", λ)
              , weights::Vector{WT}=eltype(S)[]
              , sparseTrafo=nothing
              , enforceReal::Bool=false
              , enforcePositive::Bool=false
              , iterations::Int64=10
              , kargs...) where WT

  if (reg.prox!) != (proxL2!)
    @error "CGNR only supports L2 regularizer"
  end

  M,N = size(S)
  T = eltype(S)
  cl = zeros(T,N)
  rl = zeros(T,M)     #residual vector
  zl = zeros(T,N)     #temporary vector
  pl = zeros(T,N)     #temporary vector
  vl = zeros(T,M)     #temporary vector
  xl = zeros(T,M)     #temporary vector
  αl = zero(T)        #temporary scalar
  βl = zero(T)        #temporary scalar
  ζl = zero(T)        #temporary scalar
  return CGNR(S,reg,cl,rl,zl,pl,vl,xl,αl,βl,ζl,weights,enforceReal,enforcePositive,sparseTrafo,iterations)
end

"""
init!(solver::CGNR{T,Tsparse}
              ; S::matT=solver.S
              , u::Vector{T}=T[]
              , cl::Vector{T}=T[]
              , weights::Vector{T}=solver.weights) where {T,Tsparse,matT}

(re-) initializes the CGNR iterator
"""
function init!(solver::CGNR{T,Tsparse}
              ; S::matT=solver.S
              , u::Vector{T}=T[]
              , cl::Vector{T}=T[]
              , weights::Vector{T}=solver.weights) where {T,Tsparse,matT}

  solver.S = S
  if isempty(cl)
    solver.cl[:] .= zeros(T,size(S,2))
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
    solver.zl[:] .= adjoint(S)*solver.xl
  else
    ## gemv!('C',one(T), S, rl, zero(T), zl)
    solver.zl[:] .= adjoint(S)*solver.rl
  end
  copyto!(solver.pl,solver.zl)
end

"""
    solve(solver::CGNR, u::Vector)

solves Thikhonov-regularized inverse problem using CGNR.

# Arguments
* `solver::CGNR                         - the solver containing both system matrix and regularizer
* `u::Vector`                           - data vector
* (`S::matT=solver.S`)                  - operator for the data-term of the problem
* (`startVector::Vector{T}=T[]`)        - initial guess for the solution
* (`weights::Vector{T}=solver.weights`) - weights for the data term
* (`solverInfo=nothing`)                - solverInfo for logging
"""
function solve(solver::CGNR{T,Tsparse}, u::Vector; S::matT=solver.S, startVector::Vector{T}=eltype(S)[], weights::Vector{T}=solver.weights, solverInfo=nothing, kargs...) where {T,Tsparse,matT}
  # initialize solver parameters
  init!(solver; S=S, u=u, cl=startVector, weights=weights)

  # log solver information
  solverInfo != nothing && storeInfo(solverInfo,solver.S,u,solver.cl;reg=[solver.reg])

  # perform CGNR iterations
  for (iteration, item) = enumerate(solver)
    solverInfo != nothing && storeInfo(solverInfo,solver.S,u,solver.cl;reg=[solver.reg])
  end

  return solver.cl
end


"""
  iterate(solver::CGNR{T,Tsparse}, iteration::Int=0) where {T,Tsparse}

performs one CGNR iteration.
"""
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

@inline done(solver::CGNR,iteration::Int) = iteration>=min(solver.iterations, size(solver.S,2))
