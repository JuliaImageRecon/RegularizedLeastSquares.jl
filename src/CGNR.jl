export cgnr, CGNR

mutable struct CGNR{matT,opT,vecT,T, R, PR} <: AbstractKrylovSolver
  A::matT
  AᴴA::opT
  L2::R
  constr::PR
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
  iterations::Int64
  relTol::Float64
  z0::Float64
  normalizeReg::AbstractRegularizationNormalization
end

"""
    CGNR(A, x; kargs...)

creates an `CGNR` object for the system matrix `A`.

# Arguments
* `A`                               - system matrix
* `x::vecT`                         - (optional) array with the same type and size as the solution

# Keywords
* `reg`   - regularization term vector
* `normalizeReg`         - regularization normalization scheme
* `weights::vecT=eltype(A)[]` - weights for the data term
* `AᴴA=A'*A`              - specialized normal operator, default is `A'*A`
* `iterations::Int64=10`      - number of iterations
* `relTol::Float64=eps()`         - rel tolerance for stopping criterion

See also [`createLinearSolver`](@ref), [`solve`](@ref).
"""
function CGNR(A, x::vecT=zeros(eltype(A),size(A,2)); reg::Vector{R} = [L2Regularization(zero(real(eltype(A))))]
              , weights::vecT=similar(x,0)
              , AᴴA::opT=nothing
              , iterations::Int64=10
              , relTol::Float64=eps()
              , normalizeReg::AbstractRegularizationNormalization=NoNormalization()
              , kargs...) where {opT,vecT<:AbstractVector,R<:AbstractRegularization}
            
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

  # Prepare regularization terms
  reg = normalize(CGNR, normalizeReg, reg, A, nothing)
  idx = findsink(L2Regularization, reg)
  if isnothing(idx)
    L2 = L2Regularization(zero(T))
  else
    L2 = reg[idx]
    deleteat!(reg, idx)
  end

  indices = findsinks(RealRegularization, reg)
  push!(indices, findsinks(PositiveRegularization, reg)...)
  other = [reg[i] for i in indices]
  deleteat!(reg, indices)
  if length(reg) > 0
    error("CGNR does not allow for more additional regularization terms, found $(length(reg))")
  end
  other = identity.(other)


  return CGNR(A, AᴴA,
             L2,other,cl,rl,zl,pl,vl,xl,αl,βl,ζl,
             weights,iterations,relTol,0.0,normalizeReg)
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
  solver.L2 = normalize(solver, solver.normalizeReg, solver.L2, solver.A, u)
end

"""
    solve(solver::CGNR, u; kwargs...) where vecT

solves Tikhonov-regularized inverse problem using CGNR.

# Arguments
* `solver::CGNR                         - the solver containing both system matrix and regularizer
* `u::vecT`                             - data vector

# Keywords
* `startVector::vecT=similar(u,0)`    - initial guess for the solution
* `solverInfo=nothing`                - solverInfo for logging

when a `SolverInfo` objects is passed, the residuals `solver.zl` are stored in `solverInfo.convMeas`.
"""
function solve(solver::CGNR, u;  startVector=similar(u,0), solverInfo=nothing, kargs...)
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
      for r in solver.constr
        prox!(r, solver.cl)
      end
      return nothing
    end

    mul!(solver.vl, solver.AᴴA, solver.pl)

    solver.ζl= norm(solver.zl)^2
    normvl = dot(solver.pl,solver.vl) 

    λ_ = λ(solver.L2)
    if λ_ > 0
      solver.αl = solver.ζl/(normvl+λ_*norm(solver.pl)^2)
    else
      solver.αl = solver.ζl/normvl
    end

    BLAS.axpy!(solver.αl,solver.pl,solver.cl)

    BLAS.axpy!(-solver.αl,solver.vl,solver.zl)
    
    if λ_ > 0
      BLAS.axpy!(-λ_*solver.αl,solver.pl,solver.zl)
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
