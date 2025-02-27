export SplitBregman

mutable struct SplitBregman{matT,opT,R,ropT,P,preconT} <: AbstractPrimalDualSolver
  # operators and regularization
  A::matT
  reg::Vector{R}
  regTrafo::Vector{ropT}
  proj::Vector{P}
  # fields and operators for x update
  AHA::opT
  # other parameters
  precon::preconT
  normalizeReg::AbstractRegularizationNormalization
  verbose::Bool
  iterations::Int64
  iterationsInner::Int64
  iterationsCG::Int64
  state::AbstractSolverState{<:SplitBregman}
end

mutable struct SplitBregmanState{rT <: Real, rvecT <: AbstractVector{rT}, vecT <: Union{AbstractVector{rT}, AbstractVector{Complex{rT}}}} <: AbstractSolverState{SplitBregman}
  y::vecT
  # fields and operators for x update
  β::vecT
  β_y::vecT
  # fields for primal & dual variables
  x::vecT
  z::Vector{vecT}
  zᵒˡᵈ::Vector{vecT}
  u::Vector{vecT}
  # other paremters
  ρ::rvecT
  iteration::Int64
  iter_cnt::Int64
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
end

"""
    SplitBregman(A; AHA = A'*A, precon = Identity(), reg = L1Regularization(zero(real(eltype(AHA)))), regTrafo = opEye(eltype(AHA), size(AHA,1)), normalizeReg = NoNormalization(), rho = 1e-1, iterations = 10, iterationsInner = 10, iterationsCG = 10, absTol = eps(real(eltype(AHA))), relTol = eps(real(eltype(AHA))), tolInner = 1e-5, verbose = false)
    SplitBregman( ; AHA = ,     precon = Identity(), reg = L1Regularization(zero(real(eltype(AHA)))), regTrafo = opEye(eltype(AHA), size(AHA,1)), normalizeReg = NoNormalization(), rho = 1e-1, iterations = 10, iterationsInner = 10, iterationsCG = 10, absTol = eps(real(eltype(AHA))), relTol = eps(real(eltype(AHA))), tolInner = 1e-5, verbose = false)

Creates a `SplitBregman` object for the forward operator `A` or normal operator `AHA`.

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
  * `rho::Real`                                         - weights for condition on regularized variables; can also be a vector for multiple regularization terms
  * `iterations::Int`                              - maximum number of outer iterations. Set to 1 for unconstraint split Bregman (equivalent to ADMM)
  * `iterationsInner::Int`                              - maximum number of inner iterations
  * `iterationsCG::Int`                                 - maximum number of (inner) CG iterations
  * `absTol::Real`                                      - absolute tolerance for stopping criterion
  * `relTol::Real`                                      - relative tolerance for stopping criterion
  * `tolInner::Real`                                    - relative tolerance for CG stopping criterion
  * `verbose::Bool`                                     - print residual in each iteration

This algorithm solves the constraint problem (Eq. (4.7) in [Tom Goldstein and Stanley Osher](https://doi.org/10.1137/080725891)), i.e. `||R(x)||₁` such that `||Ax -b||₂² < σ²`. In order to solve the unconstraint problem (Eq. (4.8) in [Tom Goldstein and Stanley Osher](https://doi.org/10.1137/080725891)), i.e. `||Ax -b||₂² + λ ||R(x)||₁`, you can either set `iterations=1` or use ADMM instead, which is equivalent (`iterations=1` in SplitBregman in implied in ADMM and the SplitBregman variable `iterationsInner` is simply called `iterations` in ADMM)

Like ADMM, SplitBregman differs from ISTA-type algorithms in the sense that the proximal operation is applied separately from the transformation to the space in which the penalty is applied. This is reflected by the interface which has `reg` and `regTrafo` as separate arguments. E.g., for a TV penalty, you should NOT set `reg=TVRegularization`, but instead use `reg=L1Regularization(λ), regTrafo=RegularizedLeastSquares.GradientOp(Float64; shape=(Nx,Ny,Nz))`.

See also [`createLinearSolver`](@ref), [`solve!`](@ref).
"""
SplitBregman(; AHA, kwargs...) = SplitBregman(nothing; kwargs..., AHA = AHA)

function SplitBregman(A
                    ; AHA = A'*A
                    , precon = Identity()
                    , reg = L1Regularization(zero(real(eltype(AHA))))
                    , regTrafo = opEye(eltype(AHA), size(AHA,1), S = LinearOperators.storage_type(AHA))
                    , normalizeReg::AbstractRegularizationNormalization = NoNormalization()
                    , rho = 1e-1
                    , iterations::Int = 10
                    , iterationsInner::Int = 10
                    , iterationsCG::Int = 10
                    , absTol::Real = eps(real(eltype(AHA)))
                    , relTol::Real = eps(real(eltype(AHA)))
                    , tolInner::Real = 1e-5
                    , verbose = false
                    )

  T  = eltype(AHA)
  rT = real(T)

  reg = copy(isa(reg, AbstractVector) ? reg : [reg])
  regTrafo = copy(isa(regTrafo, AbstractVector) ? regTrafo : [regTrafo])

  indices = findsinks(AbstractProjectionRegularization, reg)
  proj = [reg[i] for i in indices]
  proj = identity.(proj)
  deleteat!(reg, indices)
  #deleteat!(regTrafo, indices)

  @assert length(reg) == length(regTrafo) "reg and regTrafo must have the same length"

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
  z    = [similar(x, size(regTrafo[i],1)) for i ∈ eachindex(reg)]
  zᵒˡᵈ = [similar(z[i])                   for i ∈ eachindex(reg)]
  u    = [similar(z[i])                   for i ∈ eachindex(reg)]

  # statevariables for CG
  # we store them here to prevent CG from allocating new fields at each call
  cgStateVars = CGStateVariables(zero(x),similar(x),similar(x))

  # convergence parameters
  rᵏ   = Array{rT}(undef, length(reg))
  sᵏ   = similar(rᵏ)
  ɛᵖʳⁱ = similar(rᵏ)
  ɛᵈᵘᵃ = similar(rᵏ)


  # normalization parameters
  reg = normalize(SplitBregman, normalizeReg, reg, A, nothing)

  state = SplitBregmanState(y, β, β_y, x, z, zᵒˡᵈ, u, rho, 1, 1, cgStateVars,rᵏ,sᵏ,ɛᵖʳⁱ,ɛᵈᵘᵃ,rT(0),rT(absTol),rT(relTol),rT(tolInner))

  return SplitBregman(A,reg,regTrafo,proj,AHA,precon,normalizeReg,verbose,iterations,iterationsInner,iterationsCG,state)
end

function init!(solver::SplitBregman, state::SplitBregmanState{rT, rvecT, vecT}, b::otherT; kwargs...) where {rT, rvecT, vecT, otherT <: AbstractVector}
  y    = similar(b, size(state.y)...)
  x    = similar(b, size(state.x)...)
  β    = similar(b, size(state.β)...)
  β_y  = similar(b, size(state.β_y)...)

  z    = [similar(b, size(state.z[i])...)     for i ∈ eachindex(solver.reg)]
  zᵒˡᵈ = [similar(b, size(state.zᵒˡᵈ[i])...)  for i ∈ eachindex(solver.reg)]
  u    = [similar(b, size(state.u[i])...)     for i ∈ eachindex(solver.reg)]

  cgStateVars = CGStateVariables(zero(x),similar(x),similar(x))

  state = SplitBregmanState(y, β, β_y, x, z, zᵒˡᵈ, u, state.ρ, state.iteration, state.iter_cnt, cgStateVars,
      state.rᵏ, state.sᵏ, state.ɛᵖʳⁱ, state.ɛᵈᵘᵃ, state.σᵃᵇˢ, state.absTol, state.relTol, state.tolInner)
  
  solver.state = state
  init!(solver, state, b; kwargs...)
end

"""
  init!(solver::SplitBregman, b; x0 = 0)

(re-) initializes the SplitBregman iterator
"""
function init!(solver::SplitBregman, state::SplitBregmanState{rT, rvecT, vecT}, b::vecT; x0 = 0) where {rT, rvecT, vecT <: AbstractVector}
  state.x .= x0

  # right hand side for the x-update
  if solver.A === nothing
    state.β_y .= b
  else
    mul!(state.β_y, adjoint(solver.A), b)
  end
  state.y .= state.β_y

  # primal and dual variables
  for i ∈ eachindex(solver.reg)
    state.z[i] .= solver.regTrafo[i]*state.x
    state.u[i] .= 0
  end

  # convergence parameter
  state.rᵏ .= Inf
  state.sᵏ .= Inf
  state.ɛᵖʳⁱ .= 0
  state.ɛᵈᵘᵃ .= 0
  state.σᵃᵇˢ = sqrt(length(b)) * state.absTol

  # normalization of regularization parameters
  solver.reg = normalize(solver, solver.normalizeReg, solver.reg, solver.A, b)

  # reset interation counter
  state.iter_cnt = 1
  state.iteration = 1
end

solverconvergence(state::SplitBregmanState) = (; :primal => state.rᵏ, :dual => state.sᵏ)

function iterate(solver::SplitBregman, state::SplitBregmanState)
  if done(solver, state) return nothing end
  solver.verbose && println("SplitBregman Iteration #$(state.iteration) – Outer iteration $(state.iter_cnt)")

  # update x
  state.β .= state.β_y
  AHA = solver.AHA
  for i ∈ eachindex(solver.reg)
    mul!(state.β, adjoint(solver.regTrafo[i]), state.z[i],  state.ρ[i], 1)
    mul!(state.β, adjoint(solver.regTrafo[i]), state.u[i], -state.ρ[i], 1)
    AHA += state.ρ[i] * adjoint(solver.regTrafo[i]) * solver.regTrafo[i]
  end
  solver.verbose && println("conjugated gradients: ")
  cg!(state.x, AHA, state.β, Pl = solver.precon, maxiter = solver.iterationsCG, reltol = state.tolInner, statevars = state.cgStateVars, verbose = solver.verbose)

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
      prox!(solver.reg[i], state.z[i], λ(solver.reg[i])/state.ρ[i]) # λ is divided by 2 to match the ISTA-type algorithms
    end

    # 3. update u
    mul!(state.u[i], solver.regTrafo[i], state.x, 1, 1)
    state.u[i] .-= state.z[i]

    # update convergence criteria (one for each constraint)
    state.rᵏ[i] = norm(solver.regTrafo[i] * state.x - state.z[i])  # primal residual (x-z)
    state.sᵏ[i] = norm(state.ρ[i] * adjoint(solver.regTrafo[i]) * (state.z[i] .- state.zᵒˡᵈ[i])) # dual residual (concerning f(x))

    state.ɛᵖʳⁱ[i] = max(norm(solver.regTrafo[i] * state.x), norm(state.z[i]))
    state.ɛᵈᵘᵃ[i] = norm(state.ρ[i] * adjoint(solver.regTrafo[i]) * state.u[i])

    if solver.verbose
      println("rᵏ[$i]/ɛᵖʳⁱ[$i] = $(state.rᵏ[i]/state.ɛᵖʳⁱ[i])")
      println("sᵏ[$i]/ɛᵈᵘᵃ[$i] = $(state.sᵏ[i]/state.ɛᵈᵘᵃ[i])")
      flush(stdout)
    end
  end


  if converged(solver, state) || state.iteration >= solver.iterationsInner
    state.β_y .+= state.y
    mul!(state.β_y, solver.AHA, state.x, -1, 1)
    # reset z and b
    for i ∈ eachindex(solver.reg)
      mul!(state.z[i], solver.regTrafo[i], state.x)
      state.u[i] .= 0
    end
    state.iter_cnt += 1
    state.iteration = 0
  end

  state.iteration += 1
  return state.x, state
end

function converged(solver::SplitBregman, state)
    for i ∈ eachindex(solver.reg)
      (state.rᵏ[i] >= state.σᵃᵇˢ + state.relTol * state.ɛᵖʳⁱ[i]) && return false
      (state.sᵏ[i] >= state.σᵃᵇˢ + state.relTol * state.ɛᵈᵘᵃ[i]) && return false
    end
  return true
end

@inline done(solver::SplitBregman,state) = converged(solver, state) || (state.iteration == 1 && state.iter_cnt > solver.iterations)