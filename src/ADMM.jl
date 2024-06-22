export ADMM

mutable struct ADMM{matT,opT,R,ropT,P,preconT} <: AbstractPrimalDualSolver
  A::matT
  reg::Vector{R}
  regTrafo::Vector{ropT}
  proj::Vector{P}
  AHA::opT
  precon::preconT
  normalizeReg::AbstractRegularizationNormalization
  vary_ρ::Symbol
  verbose::Bool
  iterations::Int64
  iterationsCG::Int64
  state::AbstractSolverState{<:ADMM}
end

mutable struct ADMMState{rT <: Real, rvecT <: AbstractVector{rT}, vecT <: Union{AbstractVector{rT}, AbstractVector{Complex{rT}}}} <: AbstractSolverState{ADMM}
  # fields and operators for x update
  β::vecT
  β_y::vecT
  # fields for primal & dual variables
  x::vecT
  xᵒˡᵈ::vecT
  z::Vector{vecT}
  zᵒˡᵈ::Vector{vecT}
  u::Vector{vecT}
  uᵒˡᵈ::Vector{vecT}  
  # other paremters
  ρ::rvecT
  iteration::Int64
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
end

"""
    ADMM(A; AHA = A'*A, precon = Identity(), reg = L1Regularization(zero(real(eltype(AHA)))), regTrafo = opEye(eltype(AHA), size(AHA,1)), normalizeReg = NoNormalization(), rho = 1e-1, vary_rho = :none, iterations = 10, iterationsCG = 10, absTol = eps(real(eltype(AHA))), relTol = eps(real(eltype(AHA))), tolInner = 1e-5, verbose = false)
    ADMM( ; AHA = ,     precon = Identity(), reg = L1Regularization(zero(real(eltype(AHA)))), regTrafo = opEye(eltype(AHA), size(AHA,1)), normalizeReg = NoNormalization(), rho = 1e-1, vary_rho = :none, iterations = 10, iterationsCG = 10, absTol = eps(real(eltype(AHA))), relTol = eps(real(eltype(AHA))), tolInner = 1e-5, verbose = false)

Creates an `ADMM` object for the forward operator `A` or normal operator `AHA`.

# Required Arguments
  * `A`                                                 - forward operator
  OR
  * `AHA`                                               - normal operator (as a keyword argument)

# Optional Keyword Arguments
  * `AHA`                                               - normal operator is optional if `A` is supplied
  * `precon`                                            - preconditionner for the internal CG algorithm
  * `reg::AbstractParameterizedRegularization`          - regularization term; can also be a vector of regularization terms
  * `regTrafo`                                          - transformation to a space in which `reg` is applied; if `reg` is a vector, `regTrafo` has to be a vector of the same length. Use `opEye(eltype(AHA), size(AHA,1))` if no transformation is desired.
  * `normalizeReg::AbstractRegularizationNormalization` - regularization normalization scheme; options are `NoNormalization()`, `MeasurementBasedNormalization()`, `SystemMatrixBasedNormalization()`
  * `rho::Real`                                         - penalty of the augmented Lagrangian
  * `vary_rho::Symbol`                                  - vary rho to balance primal and dual feasibility; options `:none`, `:balance`, `:PnP`
  * `iterations::Int`                                   - maximum number of (outer) ADMM iterations
  * `iterationsCG::Int`                                 - maximum number of (inner) CG iterations
  * `absTol::Real`                                      - absolute tolerance for stopping criterion
  * `relTol::Real`                                      - relative tolerance for stopping criterion
  * `tolInner::Real`                                    - relative tolerance for CG stopping criterion
  * `verbose::Bool`                                     - print residual in each iteration

ADMM differs from ISTA-type algorithms in the sense that the proximal operation is applied separately from the transformation to the space in which the penalty is applied. This is reflected by the interface which has `reg` and `regTrafo` as separate arguments. E.g., for a TV penalty, you should NOT set `reg=TVRegularization`, but instead use `reg=L1Regularization(λ), regTrafo=RegularizedLeastSquares.GradientOp(Float64; shape=(Nx,Ny,Nz))`.

See also [`createLinearSolver`](@ref), [`solve!`](@ref).
"""
ADMM(; AHA, kwargs...) = ADMM(nothing; kwargs..., AHA = AHA)

function ADMM(A
            ; AHA = A'*A
            , precon = Identity()
            , reg = L1Regularization(zero(real(eltype(AHA))))
            , regTrafo = opEye(eltype(AHA), size(AHA,1), S = LinearOperators.storage_type(AHA))
            , normalizeReg::AbstractRegularizationNormalization = NoNormalization()
            , rho = 1e-1
            , vary_rho::Symbol = :none
            , iterations::Int = 10
            , iterationsCG::Int = 10
            , absTol::Real = eps(real(eltype(AHA)))
            , relTol::Real = eps(real(eltype(AHA)))
            , tolInner::Real = 1e-5
            , verbose = false
            )

  T  = eltype(AHA)
  rT = real(T)

  reg = isa(reg, AbstractVector) ? reg : [reg]
  regTrafo = isa(regTrafo, AbstractVector) ? regTrafo : [regTrafo]
  @assert length(reg) == length(regTrafo) "reg and regTrafo must have the same length"

  indices = findsinks(AbstractProjectionRegularization, reg)
  proj = [reg[i] for i in indices]
  proj = identity.(proj)
  deleteat!(reg, indices)
  deleteat!(regTrafo, indices)

  if typeof(rho) <: Number
    rho = [rT.(rho) for _ ∈ eachindex(reg)]
  else
    rho = rT.(rho)
  end

  x    = Vector{T}(undef, size(AHA,2))
  xᵒˡᵈ = similar(x)
  β    = similar(x)
  β_y  = similar(x)

  # fields for primal & dual variables
  z    = [similar(x, size(regTrafo[i],1)) for i ∈ eachindex(reg)]
  zᵒˡᵈ = [similar(z[i])                   for i ∈ eachindex(reg)]
  u    = [similar(z[i])                   for i ∈ eachindex(reg)]
  uᵒˡᵈ = [similar(u[i])                   for i ∈ eachindex(reg)]

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

  state = ADMMState(β, β_y, x, xᵒˡᵈ, z, zᵒˡᵈ, u, uᵒˡᵈ, rho, 0, cgStateVars, rᵏ, sᵏ, ɛᵖʳⁱ, ɛᵈᵘᵃ, rT(0), Δ, rT(absTol), rT(relTol), rT(tolInner))

  return ADMM(A, reg, regTrafo, proj, AHA, precon, normalizeReg, vary_rho, verbose, iterations, iterationsCG, state)
end

function init!(solver::ADMM, state::ADMMState{rT, rvecT, vecT}, b::otherT; kwargs...) where {rT, rvecT, vecT, otherT}
  x    = similar(b, size(state.x)...)
  xᵒˡᵈ = similar(b, size(state.xᵒˡᵈ)...)
  β    = similar(b, size(state.β)...)
  β_y  = similar(b, size(state.β_y)...)

  z    = [similar(b, size(state.z[i])...)     for i ∈ eachindex(solver.reg)]
  zᵒˡᵈ = [similar(b, size(state.zᵒˡᵈ[i])...)  for i ∈ eachindex(solver.reg)]
  u    = [similar(b, size(state.u[i])...)     for i ∈ eachindex(solver.reg)]
  uᵒˡᵈ = [similar(b, size(state.uᵒˡᵈ[i])...)  for i ∈ eachindex(solver.reg)]

  cgStateVars = CGStateVariables(zero(x),similar(x),similar(x))

  state = ADMMState(β, β_y, x, xᵒˡᵈ, z, zᵒˡᵈ, u, uᵒˡᵈ, state.ρ, state.iteration, cgStateVars,
      state.rᵏ, state.sᵏ, state.ɛᵖʳⁱ, state.ɛᵈᵘᵃ, state.σᵃᵇˢ, state.Δ, state.absTol, state.relTol, state.tolInner)
  
  solver.state = state
  init!(solver, state, b; kwargs...)
end

"""
  init!(solver::ADMM, b; x0 = 0)

(re-) initializes the ADMM iterator
"""
function init!(solver::ADMM, state::ADMMState{rT, rvecT, vecT}, b::vecT; x0 = 0) where {rT, rvecT, vecT}
  state.x .= x0

  # right hand side for the x-update
  if solver.A === nothing
    state.β_y .= b
  else
    mul!(state.β_y, adjoint(solver.A), b)
  end

  # primal and dual variables
  for i ∈ eachindex(solver.reg)
    state.z[i] .= solver.regTrafo[i] * state.x
    state.u[i] .= 0
  end

  # convergence parameter
  state.rᵏ .= Inf
  state.sᵏ .= Inf
  state.ɛᵖʳⁱ .= 0
  state.ɛᵈᵘᵃ .= 0
  state.σᵃᵇˢ = sqrt(length(b)) * state.absTol
  state.Δ .= Inf

  state.iteration = 0
  # normalization of regularization parameters
  solver.reg = normalize(solver, solver.normalizeReg, solver.reg, solver.A, b)
end

solverconvergence(solver::ADMM) = (; :primal => solver.rᵏ, :dual => solver.sᵏ)


"""
  iterate(it::ADMM, iteration::Int=0)

performs one ADMM iteration.
"""
function iterate(solver::ADMM, state::S = solver.state) where S <: AbstractSolverState{<:ADMM}
  done(solver, state) && return nothing
  solver.verbose && println("Outer ADMM Iteration #$iteration")

  # 1. solve arg min_x 1/2|| Ax-b ||² + ρ/2 Σ_i||Φi*x+ui-zi||²
  # <=> (A'A+ρ Σ_i Φi'Φi)*x = A'b+ρΣ_i Φi'(zi-ui)
  state.β .= state.β_y
  AHA = solver.AHA
  for i ∈ eachindex(solver.reg)
    mul!(state.β, adjoint(solver.regTrafo[i]), state.z[i],  state.ρ[i], 1)
    mul!(state.β, adjoint(solver.regTrafo[i]), state.u[i], -state.ρ[i], 1)
    AHA += state.ρ[i] * adjoint(solver.regTrafo[i]) * solver.regTrafo[i]
  end
  solver.verbose && println("conjugated gradients: ")
  state.xᵒˡᵈ .= state.x
  cg!(state.x, AHA, state.β, Pl=solver.precon, maxiter=solver.iterationsCG, reltol=state.tolInner, statevars=state.cgStateVars, verbose=solver.verbose)

  for proj in solver.proj
    prox!(proj, state.x)
  end

  #  proximal map for regularization terms
  for i ∈ eachindex(solver.reg)
    # swap z and zᵒˡᵈ w/o copying data
    tmp = state.zᵒˡᵈ[i]
    state.zᵒˡᵈ[i] = state.z[i]
    state.z[i] = tmp

    # 2. update z using the proximal map of 1/ρ*g(x)
    mul!(state.z[i], solver.regTrafo[i], state.x)
    state.z[i] .+= state.u[i]
    if state.ρ[i] != 0
      prox!(solver.reg[i], state.z[i], λ(solver.reg[i])/2state.ρ[i]) # λ is divided by 2 to match the ISTA-type algorithms
    end

    # 3. update u
    state.uᵒˡᵈ[i] .= state.u[i]
    mul!(state.u[i], solver.regTrafo[i], state.x, 1, 1)
    state.u[i] .-= state.z[i]

    # update convergence criteria (one for each constraint)
    state.rᵏ[i] = norm(solver.regTrafo[i] * state.x - state.z[i])  # primal residual (x-z)
    state.sᵏ[i] = norm(state.ρ[i] * adjoint(solver.regTrafo[i]) * (state.z[i] .- state.zᵒˡᵈ[i])) # dual residual (concerning f(x))

    state.ɛᵖʳⁱ[i] = max(norm(solver.regTrafo[i] * state.x), norm(state.z[i]))
    state.ɛᵈᵘᵃ[i] = norm(state.ρ[i] * adjoint(solver.regTrafo[i]) * state.u[i])

    Δᵒˡᵈ = state.Δ[i]
    state.Δ[i] = norm(state.x    .- state.xᵒˡᵈ   ) +
                  norm(state.z[i] .- state.zᵒˡᵈ[i]) +
                  norm(state.u[i] .- state.uᵒˡᵈ[i])

    if (solver.vary_ρ == :balance && state.rᵏ[i]/state.ɛᵖʳⁱ[i] > 10state.sᵏ[i]/state.ɛᵈᵘᵃ[i]) || # adapt ρ according to Boyd et al.
       (solver.vary_ρ == :PnP     && state.Δ[i]/Δᵒˡᵈ > 0.9) # adapt ρ according to Chang et al.
      state.ρ[i] *= 2
      state.u[i] ./= 2
    elseif solver.vary_ρ == :balance && state.sᵏ[i]/state.ɛᵈᵘᵃ[i] > 10state.rᵏ[i]/state.ɛᵖʳⁱ[i]
      state.ρ[i] /= 2
      state.u[i] .*= 2
    end

    if solver.verbose
      println("rᵏ[$i]/ɛᵖʳⁱ[$i] = $(solver.rᵏ[i]/solver.ɛᵖʳⁱ[i])")
      println("sᵏ[$i]/ɛᵈᵘᵃ[$i] = $(solver.sᵏ[i]/solver.ɛᵈᵘᵃ[i])")
      println("Δ[$i]/Δᵒˡᵈ[$i]  = $(solver.Δ[i]/Δᵒˡᵈ)")
      println("new ρ[$i]      = $(solver.ρ[i])")
      flush(stdout)
    end
  end

  state.iteration += 1
  return state.x, state
end

function converged(solver::ADMM, state::ADMMState)
  for i ∈ eachindex(solver.reg)
    (state.rᵏ[i] >= state.σᵃᵇˢ + state.relTol * state.ɛᵖʳⁱ[i]) && return false
    (state.sᵏ[i] >= state.σᵃᵇˢ + state.relTol * state.ɛᵈᵘᵃ[i]) && return false
  end
  return false
end

@inline done(solver::ADMM, state::ADMMState) = converged(solver, state) || state.iteration >= solver.iterations

solversolution(solver::ADMM) = solver.state.x 