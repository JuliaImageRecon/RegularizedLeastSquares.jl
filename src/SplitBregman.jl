export SplitBregman

mutable struct SplitBregman{matT,vecT,opT,R,ropT,P,rvecT,preconT,rT} <: AbstractPrimalDualSolver
  # oerators and regularization
  A::matT
  reg::Vector{R}
  regTrafo::Vector{ropT}
  proj::Vector{P}
  y::vecT
  # fields and operators for x update
  AHA::opT
  β::vecT
  β_yj::vecT
  # fields for primal & dual variables
  u::vecT
  v::Vector{vecT}
  vᵒˡᵈ::Vector{vecT}
  b::Vector{vecT}
  # other parameters
  precon::preconT
  ρ::rvecT
  iterations::Int64
  iterationsInner::Int64
  iterationsCG::Int64
  # state variables for CG
  cgStateVars::CGStateVariables
  # convergence parameters
  rk::rvecT
  sk::vecT
  eps_pri::rvecT
  eps_dt::vecT
  # eps_dual::Float64
  σᵃᵇˢ::rT
  absTol::rT
  relTol::rT
  tolInner::rT
  #counter for internal iterations
  iter_cnt::Int64
  normalizeReg::AbstractRegularizationNormalization
end

"""
    SplitBregman(A; AHA = A'*A, reg = L1Regularization(zero(eltype(AHA))), normalizeReg = NoNormalization(), precon = Identity(), rho = 1.e2absTol = eps(), relTol = eps(), tolInner = 1.e-6, iterations::Int = 10, iterationsInner::Int = 50, iterationsCG::Int = 10)

Creates a `SplitBregman` object for the forward operator `A`.

# Required Arguments
  * `A`                                                 - forward operator

# Optional Keyword Arguments
  * `AHA`                                               - normal operator is optional if `A` is supplied
  * `reg::AbstractParameterizedRegularization`          - regularization term
  * `normalizeReg::AbstractRegularizationNormalization` - regularization normalization scheme; options are `NoNormalization()`, `MeasurementBasedNormalization()`, `SystemMatrixBasedNormalization()`
  * `precon`                                            - preconditionner for the internal CG algorithm
  * `rho::Real`                                         - weights for condition on regularized variables; can also be a vector for multiple regularization terms
  * `absTol::Float64`                                   - absolute tolerance for stopping criterion
  * `relTol::Float64`                                   - relative tolerance for stopping criterion
  * `tolInner::Float64`                                 - tolerance for CG stopping criterion
  * `iterations::Int`                                   - maximum number of iterations
  * `iterationsInner::Int`                              - maximum number of inner iterations
  * `iterationsCG::Int`                                 - maximum number of CG iterations

See also [`createLinearSolver`](@ref), [`solve`](@ref).
"""
SplitBregman(; AHA = A'*A, reg = L1Regularization(zero(eltype(AHA))), normalizeReg::AbstractRegularizationNormalization = NoNormalization(), precon = Identity(), rho = 1.e2, absTol::Float64 = eps(), relTol::Float64 = eps(), tolInner::Float64 = 1.e-6, iterations::Int = 10, iterationsInner::Int = 50, iterationsCG::Int = 10) = SplitBregman(nothing; AHA, reg, normalizeReg, precon, rho, absTol, relTol, tolInner, iterations, iterationsInner, iterationsCG)

function SplitBregman(A
                    ; AHA = A'*A
                    , reg = L1Regularization(zero(eltype(AHA)))
                    , normalizeReg::AbstractRegularizationNormalization = NoNormalization()
                    , precon = Identity()
                    , rho = 1.e2
                    , absTol::Float64 = eps()
                    , relTol::Float64 = eps()
                    , tolInner::Float64 = 1.e-6
                    , iterations::Int = 10
                    , iterationsInner::Int = 50
                    , iterationsCG::Int = 10
                    )

  T  = eltype(AHA)
  rT = real(T)
  x = zeros(T,size(AHA,2))

  reg = vec(reg)

  regTrafo = []
  indices = findsinks(AbstractProjectionRegularization, reg)
  proj = [reg[i] for i in indices]
  proj = identity.(proj)
  deleteat!(reg, indices)
  # Retrieve constraint trafos
  for r in reg
    trafoReg = findfirst(ConstraintTransformedRegularization, r)
    if isnothing(trafoReg)
      push!(regTrafo, opEye(T,size(AHA,2)))
    else
      push!(regTrafo, trafoReg)
    end
  end
  regTrafo = identity.(regTrafo)

  if typeof(rho) <: Number
    rho = [rT.(rho) for _ ∈ eachindex(reg)]
  else
    rho = rT.(rho)
  end

  y    = similar(x)
  β    = similar(x)
  β_yj = similar(x)

  # fields for primal & dual variables
  u = similar(x)
  v    = [similar(x, size(AHA,2)) for i ∈ eachindex(vec(reg))]
  vᵒˡᵈ = [similar(v[i])           for i ∈ eachindex(vec(reg))]
  b    = [similar(v[i])           for i ∈ eachindex(vec(reg))]

  # statevariables for CG
  # we store them here to prevent CG from allocating new fields at each call
  statevars = CGStateVariables(zero(u),similar(u),similar(u))

  # convergence parameters
  rk = similar(x, rT, length(reg))
  sk = similar(x)
  eps_pri = similar(x, rT, length(reg))
  eps_dt = similar(x)

  iter_cnt = 1


  # normalization parameters
  reg = normalize(SplitBregman, normalizeReg, vec(reg), A, nothing)

  return SplitBregman(A,reg,regTrafo,proj,y,AHA,β,β_yj,u,v,vᵒˡᵈ,b,precon,rho,iterations,iterationsInner,iterationsCG,statevars,rk,sk,eps_pri,eps_dt,rT(0),absTol,relTol,tolInner,iter_cnt,normalizeReg)
end

"""
  init!(solver::SplitBregman, b; x0=0)

(re-) initializes the SplitBregman iterator
"""
function init!(solver::SplitBregman, b; x0=0)

  # right hand side for the x-update
  if solver.A === nothing
    solver.y .= b
  else
    mul!(solver.y, adjoint(solver.A), b)
  end
  solver.β_yj .= solver.y

  # start vector
  if any(x0 .!= 0)
    solver.u .= x0
  else
    solver.u .= solver.β_yj
  end

  # primal and dual variables
  for i ∈ eachindex(solver.reg)
    solver.v[i] .= solver.regTrafo[i]*solver.u
    solver.vᵒˡᵈ[i] .= 0
    solver.b[i] .= 0
  end

  # convergence parameter
  solver.σᵃᵇˢ = sqrt(length(b))*solver.absTol

  # normalization of regularization parameters
  solver.reg = normalize(solver, solver.normalizeReg, solver.reg, solver.A, b)

  # reset interation counter
  solver.iter_cnt = 1
end

"""
    solve(solver::SplitBregman, b; tartVector::vecT=similar(b,0), solverInfo=nothing)

solves an inverse problem using the Split Bregman method.

# Arguments
* `solver::SplitBregman`              - the solver containing both system matrix and regularizer
* `b::vecT`                           - data vector

# Keywords
* `x0::vecT=similar(b,0)`  - initial guess for the solution
* `solverInfo=nothing`              - solverInfo for logging

when a `SolverInfo` objects is passed, the primal residuals `solver.rk`
and the dual residual `norm(solver.sk)` are stored in `solverInfo.convMeas`.
"""
function solve(solver::SplitBregman, b; x0=0, solverInfo=nothing)
  # initialize solver parameters
  init!(solver, b; x0=x0)

  # log solver information
  solverInfo != nothing && storeInfo(solverInfo,solver.v,solver.rk...,norm(solver.sk))

  # perform SplitBregman iterations
  for (iteration, item) = enumerate(solver)
    solverInfo != nothing && storeInfo(solverInfo,solver.v,solver.rk...,norm(solver.sk))
  end

  return solver.u
end


function iterate(solver::SplitBregman, iteration=1)
  if done(solver, iteration) return nothing end

  # update u
  solver.β .= solver.β_yj
  AHA = solver.AHA
  for i ∈ eachindex(solver.reg)
    solver.β .+= solver.ρ[i] * adjoint(solver.regTrafo[i]) * (solver.v[i].-solver.b[i])
    AHA       += solver.ρ[i] * adjoint(solver.regTrafo[i]) * solver.regTrafo[i]
  end
  cg!(solver.u, AHA, solver.β, Pl = solver.precon, maxiter = solver.iterationsCG, reltol=solver.tolInner)

  for proj in solver.proj
    prox!(proj, solver.u)
  end

  #  proximal map for regularization terms
  for i ∈ eachindex(solver.reg)
    copyto!(solver.vᵒˡᵈ[i], solver.v[i])
    solver.v[i][:] .= solver.regTrafo[i]*solver.u .+ solver.b[i]
    if solver.ρ[i] != 0
      prox!(solver.reg[i], solver.v[i], λ(solver.reg[i])/solver.ρ[i])
    end
  end

  # update b
  for i ∈ eachindex(solver.reg)
    solver.b[i] .+= solver.regTrafo[i]*solver.u .- solver.v[i]
  end

  # update convergence criteria
  # primal residuals norms (one for each constraint)
  for i ∈ eachindex(solver.reg)
    solver.rk[i] = norm(solver.regTrafo[i]*solver.u-solver.v[i])
    solver.eps_pri[i] = solver.σᵃᵇˢ + solver.relTol*max( norm(solver.regTrafo[i]*solver.u), norm(solver.v[i]) )
  end
  # accumulated dual residual
  # effectively this corresponds to combining all constraints into one larger constraint.
  solver.sk .= 0
  solver.eps_dt .= 0
  for i ∈ eachindex(solver.reg)
    solver.sk[:] .+= solver.ρ[i]*adjoint(solver.regTrafo[i])*(solver.v[i].-solver.vᵒˡᵈ[i])
    solver.eps_dt[:] .+= solver.ρ[i]*adjoint(solver.regTrafo[i])*solver.b[i]
  end

  if update_y(solver,iteration)
    solver.β_yj .+= solver.y .- solver.AHA * solver.u
    # reset v and b
    for i ∈ eachindex(solver.reg)
      solver.v[i] .= solver.regTrafo[i]*solver.u
      solver.b[i] .= 0
    end
    solver.iter_cnt += 1
    iteration = 0
  end

  return solver.rk[1], iteration+1

end

function converged(solver::SplitBregman)
  if norm(solver.sk) >= solver.σᵃᵇˢ+solver.relTol*norm(solver.eps_dt)
    return false
  else
    for i=1:length(solver.reg)
      (solver.rk[i] >= solver.eps_pri[i]) && return false
    end
  end

  return true
end

@inline done(solver::SplitBregman,iteration::Int) = (iteration==1 && solver.iter_cnt>solver.iterations)

function update_y(solver::SplitBregman,iteration::Int)
  conv = converged(solver)
  return conv || iteration >= solver.iterationsInner
end
