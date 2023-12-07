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
  β_y::vecT
  # fields for primal & dual variables
  x::vecT
  z::Vector{vecT}
  zᵒˡᵈ::Vector{vecT}
  u::Vector{vecT}
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
  verbose::Bool
end

"""
    SplitBregman(A; AHA = A'*A, reg = L1Regularization(zero(eltype(AHA))), normalizeReg = NoNormalization(), precon = Identity(), rho = 1.e2absTol = eps(), relTol = eps(), tolInner = 1.e-6, iterations::Int = 10, iterationsInner::Int = 50, iterationsCG::Int = 10, verbose = false)
    SplitBregman( ; AHA = A'*A, reg = L1Regularization(zero(eltype(AHA))), normalizeReg = NoNormalization(), precon = Identity(), rho = 1.e2absTol = eps(), relTol = eps(), tolInner = 1.e-6, iterations::Int = 10, iterationsInner::Int = 50, iterationsCG::Int = 10, verbose = false)

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
  * `verbose::Bool`                                     - print residual in each iteration

See also [`createLinearSolver`](@ref), [`solve!`](@ref).
"""
SplitBregman(; AHA = A'*A, kwargs...) = SplitBregman(nothing; kwargs..., AHA = AHA)

function SplitBregman(A
                    ; AHA = A'*A
                    , reg = L1Regularization(zero(eltype(AHA)))
                    , normalizeReg::AbstractRegularizationNormalization = NoNormalization()
                    , precon = Identity()
                    , rho = 1.e2
                    , absTol = eps()
                    , relTol = eps()
                    , tolInner = 1.e-6
                    , iterations::Int = 10
                    , iterationsInner::Int = 50
                    , iterationsCG::Int = 10
                    , verbose = false
                    )

  T  = eltype(AHA)
  rT = real(T)

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

  x   = Vector{T}(undef,size(AHA,2))
  y   = similar(x)
  β   = similar(x)
  β_y = similar(x)

  # fields for primal & dual variables
  z    = [similar(x, size(AHA,2)) for i ∈ eachindex(vec(reg))]
  zᵒˡᵈ = [similar(z[i])           for i ∈ eachindex(vec(reg))]
  u    = [similar(z[i])           for i ∈ eachindex(vec(reg))]

  # statevariables for CG
  # we store them here to prevent CG from allocating new fields at each call
  cgStateVars = CGStateVariables(zero(x),similar(x),similar(x))

  # convergence parameters
  rk = similar(x, rT, length(reg))
  sk = similar(x)
  eps_pri = similar(x, rT, length(reg))
  eps_dt = similar(x)

  iter_cnt = 1


  # normalization parameters
  reg = normalize(SplitBregman, normalizeReg, vec(reg), A, nothing)

  return SplitBregman(A,reg,regTrafo,proj,y,AHA,β,β_y,x,z,zᵒˡᵈ,u,precon,rho,iterations,iterationsInner,iterationsCG,cgStateVars,rk,sk,eps_pri,eps_dt,rT(0),rT(absTol),rT(relTol),rT(tolInner),iter_cnt,normalizeReg,verbose)
end

"""
  init!(solver::SplitBregman, b; x0 = 0)

(re-) initializes the SplitBregman iterator
"""
function init!(solver::SplitBregman, b; x0 = 0)
  solver.x .= x0

  # right hand side for the x-update
  if solver.A === nothing
    solver.β_y .= b
  else
    mul!(solver.β_y, adjoint(solver.A), b)
  end
  solver.y .= solver.β_y

  # primal and dual variables
  for i ∈ eachindex(solver.reg)
    solver.z[i] .= solver.regTrafo[i]*solver.x
    solver.u[i] .= 0
  end

  # convergence parameter
  solver.σᵃᵇˢ = sqrt(length(b))*solver.absTol

  # normalization of regularization parameters
  solver.reg = normalize(solver, solver.normalizeReg, solver.reg, solver.A, b)

  # reset interation counter
  solver.iter_cnt = 1
end

solverconvergence(solver::SplitBregman) = (; :primal => solver.rk, :dual => norm(solver.sk))

function iterate(solver::SplitBregman, iteration=1)
  if done(solver, iteration) return nothing end

  # update x
  solver.β .= solver.β_y
  AHA = solver.AHA
  for i ∈ eachindex(solver.reg)
    mul!(solver.β, adjoint(solver.regTrafo[i]), solver.z[i],  solver.ρ[i], 1)
    mul!(solver.β, adjoint(solver.regTrafo[i]), solver.u[i], -solver.ρ[i], 1)
    AHA += solver.ρ[i] * adjoint(solver.regTrafo[i]) * solver.regTrafo[i]
  end
  solver.verbose && println("conjugated gradients: ")
  cg!(solver.x, AHA, solver.β, Pl = solver.precon, maxiter = solver.iterationsInner, reltol=solver.tolInner, statevars=solver.cgStateVars, verbose = solver.verbose)

  for proj in solver.proj
    prox!(proj, solver.x)
  end

  #  proximal map for regularization terms
  for i ∈ eachindex(solver.reg)
    # swap z and zᵒˡᵈ w/o copying data
    tmp = solver.zᵒˡᵈ[i]
    solver.zᵒˡᵈ[i] = solver.z[i]
    solver.z[i] = tmp

    # 2. update z using the proximal map of 1/ρ*g(x)
    mul!(solver.z[i], solver.regTrafo[i], solver.x)
    solver.z[i] .+= solver.u[i]
    if solver.ρ[i] != 0
      prox!(solver.reg[i], solver.z[i], λ(solver.reg[i])/solver.ρ[i])
    end

    # 3. update u
    mul!(solver.u[i], solver.regTrafo[i], solver.x, 1, 1)
    solver.u[i] .-= solver.z[i]

    # update convergence criteria
    # primal residuals norms (one for each constraint)
    solver.rk[i] = norm(solver.regTrafo[i] * solver.x - solver.z[i])
    solver.eps_pri[i] = solver.σᵃᵇˢ + solver.relTol * max(norm(solver.regTrafo[i]*solver.x), norm(solver.z[i]))
  end

  # accumulated dual residual
  # effectively this corresponds to combining all constraints into one larger constraint.
  solver.sk .= 0
  solver.eps_dt .= 0
  for i ∈ eachindex(solver.reg)
    mul!(solver.sk,     adjoint(solver.regTrafo[i]), solver.z[i],     solver.ρ[i], 1)
    mul!(solver.sk,     adjoint(solver.regTrafo[i]), solver.zᵒˡᵈ[i], -solver.ρ[i], 1)
    mul!(solver.eps_dt, adjoint(solver.regTrafo[i]), solver.u[i],     solver.ρ[i], 1)
  end


  if converged(solver) || iteration >= solver.iterationsInner
    solver.β_y .+= solver.y
    mul!(solver.β_y, solver.AHA, solver.x, -1, 1)
    # reset z and b
    for i ∈ eachindex(solver.reg)
      mul!(solver.z[i], solver.regTrafo[i], solver.x)
      solver.u[i] .= 0
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