export SplitBregman

mutable struct SplitBregman{matT,opT,R,ropT,P,vecT,rvecT,preconT,rT} <: AbstractPrimalDualSolver
  # operators and regularization
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
  rᵏ::rvecT
  sᵏ::rvecT
  ɛᵖʳⁱ::rvecT
  ɛᵈᵘᵃ::rvecT
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
    SplitBregman(A; AHA = A'*A, precon = Identity(), reg = L1Regularization(zero(eltype(AHA))), normalizeReg = NoNormalization(), rho = 1e-1, iterations = 1, iterationsInner = 10, iterationsCG = 10, absTol = eps(real(eltype(AHA))), relTol = eps(real(eltype(AHA))), tolInner = 1e-5, verbose = false)
    SplitBregman( ; AHA = ,     precon = Identity(), reg = L1Regularization(zero(eltype(AHA))), normalizeReg = NoNormalization(), rho = 1e-1, iterations = 1, iterationsInner = 10, iterationsCG = 10, absTol = eps(real(eltype(AHA))), relTol = eps(real(eltype(AHA))), tolInner = 1e-5, verbose = false)

Creates a `SplitBregman` object for the forward operator `A` or normal operator `AHA`.

# Required Arguments
  * `A`                                                 - forward operator
  OR
  * `AHA`                                               - normal operator (as a keyword argument)

# Optional Keyword Arguments
  * `AHA`                                               - normal operator is optional if `A` is supplied
  * `precon`                                            - preconditionner for the internal CG algorithm
  * `reg::AbstractParameterizedRegularization`          - regularization term; can also be a vector of regularization terms
  * `normalizeReg::AbstractRegularizationNormalization` - regularization normalization scheme; options are `NoNormalization()`, `MeasurementBasedNormalization()`, `SystemMatrixBasedNormalization()`
  * `rho::Real`                                         - weights for condition on regularized variables; can also be a vector for multiple regularization terms
  * `iterations::Int`                                   - maximum number of outer iterations. Set to 1 for unconstraint split Bregman
  * `iterationsInner::Int`                              - maximum number of inner iterations
  * `iterationsCG::Int`                                 - maximum number of (inner) CG iterations
  * `absTol::Real`                                      - absolute tolerance for stopping criterion
  * `relTol::Real`                                      - relative tolerance for stopping criterion
  * `tolInner::Real`                                    - relative tolerance for CG stopping criterion
  * `verbose::Bool`                                     - print residual in each iteration

See also [`createLinearSolver`](@ref), [`solve!`](@ref).
"""
SplitBregman(; AHA, kwargs...) = SplitBregman(nothing; kwargs..., AHA = AHA)

function SplitBregman(A
                    ; AHA = A'*A
                    , precon = Identity()
                    , reg = L1Regularization(zero(eltype(AHA)))
                    , normalizeReg::AbstractRegularizationNormalization = NoNormalization()
                    , rho = 1e-1
                    , iterations::Int = 1
                    , iterationsInner::Int = 10
                    , iterationsCG::Int = 10
                    , absTol::Real = eps(real(eltype(AHA)))
                    , relTol::Real = eps(real(eltype(AHA)))
                    , tolInner::Real = 1e-5
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
      push!(regTrafo, trafoReg.trafo)
    end
  end
  regTrafo = identity.(regTrafo)

  if typeof(rho) <: Number
    rho = [rT.(rho) for _ ∈ eachindex(reg)]
  else
    rho = rT.(rho)
  end

  x   = Vector{T}(undef, size(AHA,2))
  y   = similar(x)
  β   = similar(x)
  β_y = similar(x)

  # fields for primal & dual variables
  z    = [similar(x, size(regTrafo[i],1)) for i ∈ eachindex(vec(reg))]
  zᵒˡᵈ = [similar(z[i])                   for i ∈ eachindex(vec(reg))]
  u    = [similar(z[i])                   for i ∈ eachindex(vec(reg))]

  # statevariables for CG
  # we store them here to prevent CG from allocating new fields at each call
  cgStateVars = CGStateVariables(zero(x),similar(x),similar(x))

  # convergence parameters
  rᵏ   = Array{rT}(undef, length(reg))
  sᵏ   = similar(rᵏ)
  ɛᵖʳⁱ = similar(rᵏ)
  ɛᵈᵘᵃ = similar(rᵏ)

  iter_cnt = 1


  # normalization parameters
  reg = normalize(SplitBregman, normalizeReg, reg, A, nothing)

  return SplitBregman(A,reg,regTrafo,proj,y,AHA,β,β_y,x,z,zᵒˡᵈ,u,precon,rho,iterations,iterationsInner,iterationsCG,cgStateVars,rᵏ,sᵏ,ɛᵖʳⁱ,ɛᵈᵘᵃ,rT(0),rT(absTol),rT(relTol),rT(tolInner),iter_cnt,normalizeReg,verbose)
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
  solver.rᵏ .= Inf
  solver.sᵏ .= Inf
  solver.ɛᵖʳⁱ .= 0
  solver.ɛᵈᵘᵃ .= 0
  solver.σᵃᵇˢ = sqrt(length(b)) * solver.absTol

  # normalization of regularization parameters
  solver.reg = normalize(solver, solver.normalizeReg, solver.reg, solver.A, b)

  # reset interation counter
  solver.iter_cnt = 1
end

solverconvergence(solver::SplitBregman) = (; :primal => solver.rᵏ, :dual => solver.sᵏ)

function iterate(solver::SplitBregman, iteration=1)
  if done(solver, iteration) return nothing end
  solver.verbose && println("SplitBregman Iteration #$iteration – Outer iteration $(solver.iter_cnt)")

  # update x
  solver.β .= solver.β_y
  AHA = solver.AHA
  for i ∈ eachindex(solver.reg)
    mul!(solver.β, adjoint(solver.regTrafo[i]), solver.z[i],  solver.ρ[i]/λ(solver.reg[i]), 1)
    mul!(solver.β, adjoint(solver.regTrafo[i]), solver.u[i], -solver.ρ[i]/λ(solver.reg[i]), 1)
    AHA += solver.ρ[i]/λ(solver.reg[i]) * adjoint(solver.regTrafo[i]) * solver.regTrafo[i]
  end
  solver.verbose && println("conjugated gradients: ")
  cg!(solver.x, AHA, solver.β, Pl = solver.precon, maxiter = solver.iterationsCG, reltol = solver.tolInner, statevars = solver.cgStateVars, verbose = solver.verbose)

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

    # update convergence criteria (one for each constraint)
    solver.rᵏ[i] = norm(solver.regTrafo[i] * solver.x - solver.z[i])  # primal residual (x-z)
    solver.sᵏ[i] = norm(solver.ρ[i] * adjoint(solver.regTrafo[i]) * (solver.z[i] .- solver.zᵒˡᵈ[i])) # dual residual (concerning f(x))

    solver.ɛᵖʳⁱ[i] = max(norm(solver.regTrafo[i] * solver.x), norm(solver.z[i]))
    solver.ɛᵈᵘᵃ[i] = norm(solver.ρ[i] * adjoint(solver.regTrafo[i]) * solver.u[i])

    if solver.verbose
      println("rᵏ[$i]   = $(solver.rᵏ[i])")
      println("sᵏ[$i]   = $(solver.sᵏ[i])")
      println("ɛᵖʳⁱ[$i] = $(solver.ɛᵖʳⁱ[i])")
      println("ɛᵈᵘᵃ[$i] = $(solver.ɛᵈᵘᵃ[i])")
      flush(stdout)
    end
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

  return solver.rᵏ, iteration+1
end

function converged(solver::SplitBregman)
    for i ∈ eachindex(solver.reg)
      (solver.rᵏ[i] >= solver.σᵃᵇˢ + solver.relTol * solver.ɛᵖʳⁱ[i]) && return false
      (solver.sᵏ[i] >= solver.σᵃᵇˢ + solver.relTol * solver.ɛᵈᵘᵃ[i]) && return false
    end
  return true
end

@inline done(solver::SplitBregman,iteration::Int) = converged(solver) || (iteration == 1 && solver.iter_cnt > solver.iterations)