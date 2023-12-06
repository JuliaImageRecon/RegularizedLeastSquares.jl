export ADMM

mutable struct ADMM{rT,matT,opT,R,ropT,P,vecT,rvecT,preconT} <: AbstractPrimalDualSolver where {vecT <: AbstractVector{Union{rT, Complex{rT}}}, rvecT <: AbstractVector{rT}}
  # operators and regularization
  A::matT
  reg::Vector{R}
  regTrafo::Vector{ropT}
  proj::Vector{P}
  # fields and operators for x update
  AHA::opT
  β::vecT
  β_y::vecT
  # fields for primal & dual variables
  x::vecT
  xᵒˡᵈ::vecT
  z::Vector{vecT}
  zᵒˡᵈ::Vector{vecT}
  u::Vector{vecT}
  uᵒˡᵈ::Vector{vecT}
  # other parameters
  precon::preconT
  ρ::rvecT # TODO: Switch all these vectors to Tuple
  iterations::Int64
  iterationsCG::Int64
  # state variables for CG
  cgStateVars::CGStateVariables
  # convergence parameters
  rᵏ::rvecT
  sᵏ::rvecT
  ɛᵖʳⁱ::rvecT
  ɛᵈᵘᵃ::rvecT
  σᵃᵇˢ::rT
  Δ::rvecT
  absTol::rT
  relTol::rT
  tolInner::rT
  normalizeReg::AbstractRegularizationNormalization
  vary_ρ::Symbol
  verbose::Bool
end

"""
    ADMM(A; AHA = A'*A, precon = Identity(), reg = L1Regularization(zero(eltype(AHA))), normalizeReg = NoNormalization(), rho = 1e-1, vary_rho = :none, iterations = 50, iterationsCG = 10, absTol = eps(real(eltype(AHA))), relTol = eps(real(eltype(AHA))), tolInner = 1e-5, verbose = false)
    ADMM( ; AHA = ,     precon = Identity(), reg = L1Regularization(zero(eltype(AHA))), normalizeReg = NoNormalization(), rho = 1e-1, vary_rho = :none, iterations = 50, iterationsCG = 10, absTol = eps(real(eltype(AHA))), relTol = eps(real(eltype(AHA))), tolInner = 1e-5, verbose = false)

creates an `ADMM` object for the forward operator `A` or normal operator `AHA`.

# Required Arguments
  * `A`                                                 - forward operator
  OR
  * `AHA`                                               - normal operator (as a keyword argument)

# Optional Keyword Arguments
  * `AHA`                                               - normal operator is optional if `A` is supplied
  * `precon`                                            - preconditionner for the internal CG algorithm
  * `reg::AbstractParameterizedRegularization`          - regularization term; can also be a vector of regularization terms
  * `normalizeReg::AbstractRegularizationNormalization` - regularization normalization scheme; options are `NoNormalization()`, `MeasurementBasedNormalization()`, `SystemMatrixBasedNormalization()`
  * `rho::Real`                                         - penalty of the augmented Lagrangian
  * `vary_rho::Symbol`                                  - vary rho to balance primal and dual feasibility; options `:none`, `:balance`, `:PnP`
  * `iterations::Int`                                   - maximum number of (outer) ADMM iterations
  * `iterationsCG::Int`                                 - max number of (inner) CG iterations
  * `absTol::Real`                                      - abs tolerance for stopping criterion
  * `relTol::Real`                                      - tolerance for stopping criterion
  * `tolInner::Real`                                    - rel tolerance for CG stopping criterion
  * `verbose::Bool`                                     - print residual in each iteration

See also [`createLinearSolver`](@ref), [`solve`](@ref).
"""
ADMM(; AHA = A'*A, precon = Identity(), reg = L1Regularization(zero(eltype(AHA))), normalizeReg::AbstractRegularizationNormalization = NoNormalization(), rho = 1e-1, vary_rho::Symbol = :none, iterations::Int = 50, iterationsCG::Int = 10, absTol::Real = eps(real(eltype(AHA))), relTol::Real = eps(real(eltype(AHA))), tolInner::Real = 1e-5, verbose = false) = ADMM(nothing; AHA, precon, reg, normalizeReg, rho, vary_rho, iterations, iterationsCG, absTol, relTol, tolInner, verbose)

function ADMM(A
            ; AHA = A'*A
            , precon = Identity()
            , reg = L1Regularization(zero(eltype(AHA)))
            , normalizeReg::AbstractRegularizationNormalization = NoNormalization()
            , rho = 1e-1
            , vary_rho::Symbol = :none
            , iterations::Int = 50
            , iterationsCG::Int = 10
            , absTol::Real = eps(real(eltype(AHA)))
            , relTol::Real = eps(real(eltype(AHA)))
            , tolInner::Real = 1e-5
            , verbose = false
            )
  # TODO: The constructor is not type stable

  T  = eltype(AHA)
  rT = real(T)

  reg = vec(reg) # using a custom method of vec(.)

  regTrafo = []
  indices = findsinks(AbstractProjectionRegularization, reg)
  proj = [reg[i] for i in indices]
  proj = identity.(proj)
  deleteat!(reg, indices)
  # Retrieve constraint trafos
  for r in reg
    trafoReg = findfirst(ConstraintTransformedRegularization, r)
    if isnothing(trafoReg)
      push!(regTrafo, opEye(eltype(AHA),size(AHA,2)))
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

  x    = Vector{T}(undef,size(AHA,2))
  xᵒˡᵈ = similar(x)
  β    = similar(x)
  β_y  = similar(x)

  # fields for primal & dual variables
  z    = [similar(x, size(AHA,2)) for i=1:length(reg)]
  zᵒˡᵈ = [similar(z[i]) for i=1:length(reg)]
  u    = [similar(z[i]) for i=1:length(reg)]
  uᵒˡᵈ = [similar(u[i]) for i=1:length(reg)]


  # statevariables for CG
  # we store them here to prevent CG from allocating new fields at each call
  cgStateVars = CGStateVariables(zero(x),similar(x),similar(x))

  # convergence parameters
  rᵏ   = Array{rT}(undef, length(reg))
  sᵏ   = similar(rᵏ)
  ɛᵖʳⁱ = similar(rᵏ)
  ɛᵈᵘᵃ = similar(rᵏ)
  Δ    = similar(rᵏ)

  # normalization parameters
  reg = normalize(ADMM, normalizeReg, reg, A, nothing)

  return ADMM(A,reg,regTrafo,proj,AHA,β,β_y,x,xᵒˡᵈ,z,zᵒˡᵈ,u,uᵒˡᵈ,precon,rho,iterations
              ,iterationsCG,cgStateVars, rᵏ,sᵏ,ɛᵖʳⁱ,ɛᵈᵘᵃ,zero(rT),Δ,rT(absTol),rT(relTol),rT(tolInner)
              ,normalizeReg, vary_rho, verbose)
end

"""
  init!(solver::ADMM, b; x=similar(b,0))

(re-) initializes the ADMM iterator
"""
function init!(solver::ADMM, b::AbstractVector{T}; x0=0) where T
  solver.x .= x0

  # right hand side for the x-update
  if solver.A === nothing
    solver.β_y .= b
  else
    mul!(solver.β_y, adjoint(solver.A), b)
  end

  # primal and dual variables
  for i=1:length(solver.reg)
    solver.z[i] .= solver.regTrafo[i]*solver.x
    solver.u[i] .= 0
  end

  # convergence parameter
  solver.rᵏ .= Inf
  solver.sᵏ .= Inf
  solver.ɛᵖʳⁱ .= 0
  solver.ɛᵈᵘᵃ .= 0
  solver.σᵃᵇˢ = sqrt(length(b))*solver.absTol
  solver.Δ .= Inf

  # normalization of regularization parameters
  solver.reg = normalize(solver, solver.normalizeReg, solver.reg, solver.A, b)
end


"""
  iterate(it::ADMM, iteration::Int=0)

performs one ADMM iteration.
"""
function iterate(solver::ADMM, iteration=0)
  if done(solver, iteration) return nothing end
  solver.verbose && println("Outer ADMM Iteration #$iteration")

  # 1. solve arg min_x 1/2|| Ax-b ||² + ρ/2 Σ_i||Φi*x+ui-zi||²
  # <=> (A'A+ρ Σ_i Φi'Φi)*x = A'b+ρΣ_i Φi'(zi-ui)
  solver.β .= solver.β_y
  AHA = solver.AHA
  for i ∈ eachindex(solver.reg)
    mul!(solver.β, adjoint(solver.regTrafo[i]), solver.z[i],  solver.ρ[i], 1)
    mul!(solver.β, adjoint(solver.regTrafo[i]), solver.u[i], -solver.ρ[i], 1)
    AHA       += solver.ρ[i] * adjoint(solver.regTrafo[i]) * solver.regTrafo[i]
  end
  solver.verbose && println("conjugated gradients: ")
  solver.xᵒˡᵈ .= solver.x
  cg!(solver.x, AHA, solver.β, Pl=solver.precon, maxiter=solver.iterationsCG, reltol=solver.tolInner, statevars=solver.cgStateVars, verbose = solver.verbose)

  for proj in solver.proj
    prox!(proj, solver.x)
  end

  for i ∈ eachindex(solver.reg)
    # swap v and vᵒˡᵈ w/o copying data
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
    solver.uᵒˡᵈ[i] .= solver.u[i]
    mul!(solver.u[i], solver.regTrafo[i], solver.x, 1, 1)
    solver.u[i] .-= solver.z[i]

    # update convergence measures (one for each constraint)
    solver.rᵏ[i] = norm(solver.regTrafo[i]*solver.x-solver.z[i])  # primal residual (x-z)
    solver.sᵏ[i] = norm(solver.ρ[i] * adjoint(solver.regTrafo[i]) * (solver.z[i] .- solver.zᵒˡᵈ[i])) # dual residual (concerning f(x))

    solver.ɛᵖʳⁱ[i] = max(norm(solver.regTrafo[i]*solver.x), norm(solver.z[i]))
    solver.ɛᵈᵘᵃ[i] = norm(solver.ρ[i] * adjoint(solver.regTrafo[i]) * solver.u[i])

    Δᵒˡᵈ = solver.Δ[i]
    solver.Δ[i] = norm(solver.x    .- solver.xᵒˡᵈ   ) +
                  norm(solver.z[i] .- solver.zᵒˡᵈ[i]) +
                  norm(solver.u[i] .- solver.uᵒˡᵈ[i])

    if (solver.vary_ρ == :balance && solver.rᵏ[i]/solver.ɛᵖʳⁱ[i] > 10solver.sᵏ[i]/solver.ɛᵈᵘᵃ[i]) || # adapt ρ according to Boyd et al.
       (solver.vary_ρ == :PnP     && solver.Δ[i]/Δᵒˡᵈ > 0.9) # adapt ρ according to Chang et al.
      solver.ρ[i] *= 2
      solver.u[i] ./= 2
    elseif solver.vary_ρ == :balance && solver.sᵏ[i]/solver.ɛᵈᵘᵃ[i] > 10solver.rᵏ[i]/solver.ɛᵖʳⁱ[i]
      solver.ρ[i] /= 2
      solver.u[i] .*= 2
    end

    if solver.verbose
      println("rᵏ[$i] = $(solver.rᵏ[i])")
      println("sᵏ[$i] = $(solver.sᵏ[i])")
      println("ɛᵖʳⁱ[$i] = $(solver.ɛᵖʳⁱ[i])")
      println("ɛᵈᵘᵃ[$i] = $(solver.ɛᵈᵘᵃ[i])")
      println("Δᵒˡᵈ = $(Δᵒˡᵈ)")
      println("Δ[$i] = $(solver.Δ[i])")
      println("Δ/Δᵒˡᵈ = $(solver.Δ[i]/Δᵒˡᵈ)")
      println("current ρ[$i] = $(solver.ρ[i])")
      flush(stdout)
    end
  end

  # return the primal feasibility measure as item and iteration number as state
  return solver.rᵏ, iteration+1
end

function converged(solver::ADMM)
  for i=1:length(solver.reg)
    (solver.rᵏ[i] >= solver.σᵃᵇˢ + solver.relTol * solver.ɛᵖʳⁱ[i]) && return false
    (solver.sᵏ[i] >= solver.σᵃᵇˢ + solver.relTol * solver.ɛᵈᵘᵃ[i]) && return false
  end
  return true
end

@inline done(solver::ADMM,iteration::Int) = converged(solver) || iteration>=solver.iterations
