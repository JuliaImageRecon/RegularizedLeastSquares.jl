export ADMM

mutable struct ADMM{matT,opT,R,ropT,P,vecT,rvecT,preconT,rT} <: AbstractPrimalDualSolver where {vecT <: AbstractVector{Union{rT, Complex{rT}}}, rvecT <: AbstractVector{rT}}
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
  ρ::rvecT
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
            , regTrafo = opEye(eltype(AHA), size(AHA,1))
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

  return ADMM(A, reg, regTrafo, proj, AHA, β, β_y, x, xᵒˡᵈ, z, zᵒˡᵈ, u, uᵒˡᵈ, precon, rho, iterations, iterationsCG, cgStateVars, rᵏ, sᵏ, ɛᵖʳⁱ, ɛᵈᵘᵃ, rT(0), Δ, rT(absTol), rT(relTol), rT(tolInner), normalizeReg, vary_rho, verbose)
end

"""
  init!(solver::ADMM, b; x0 = 0)

(re-) initializes the ADMM iterator
"""
function init!(solver::ADMM, b; x0=0)
  solver.x .= x0

  # right hand side for the x-update
  if solver.A === nothing
    solver.β_y .= b
  else
    mul!(solver.β_y, adjoint(solver.A), b)
  end

  # primal and dual variables
  for i ∈ eachindex(solver.reg)
    solver.z[i] .= solver.regTrafo[i] * solver.x
    solver.u[i] .= 0
  end

  # convergence parameter
  solver.rᵏ .= Inf
  solver.sᵏ .= Inf
  solver.ɛᵖʳⁱ .= 0
  solver.ɛᵈᵘᵃ .= 0
  solver.σᵃᵇˢ = sqrt(length(b)) * solver.absTol
  solver.Δ .= Inf

  # normalization of regularization parameters
  solver.reg = normalize(solver, solver.normalizeReg, solver.reg, solver.A, b)
end

solverconvergence(solver::ADMM) = (; :primal => solver.rᵏ, :dual => solver.sᵏ)


"""
  iterate(it::ADMM, iteration::Int=0)

performs one ADMM iteration.
"""
function iterate(solver::ADMM, iteration=1)
  done(solver, iteration) && return nothing
  solver.verbose && println("Outer ADMM Iteration #$iteration")

  # 1. solve arg min_x 1/2|| Ax-b ||² + ρ/2 Σ_i||Φi*x+ui-zi||²
  # <=> (A'A+ρ Σ_i Φi'Φi)*x = A'b+ρΣ_i Φi'(zi-ui)
  solver.β .= solver.β_y
  AHA = solver.AHA
  for i ∈ eachindex(solver.reg)
    mul!(solver.β, adjoint(solver.regTrafo[i]), solver.z[i],  solver.ρ[i], 1)
    mul!(solver.β, adjoint(solver.regTrafo[i]), solver.u[i], -solver.ρ[i], 1)
    AHA += solver.ρ[i] * adjoint(solver.regTrafo[i]) * solver.regTrafo[i]
  end
  solver.verbose && println("conjugated gradients: ")
  solver.xᵒˡᵈ .= solver.x
  cg!(solver.x, AHA, solver.β, Pl=solver.precon, maxiter=solver.iterationsCG, reltol=solver.tolInner, statevars=solver.cgStateVars, verbose=solver.verbose)

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
      prox!(solver.reg[i], solver.z[i], λ(solver.reg[i])/2solver.ρ[i]) # λ is divided by 2 to match the ISTA-type algorithms
    end

    # 3. update u
    solver.uᵒˡᵈ[i] .= solver.u[i]
    mul!(solver.u[i], solver.regTrafo[i], solver.x, 1, 1)
    solver.u[i] .-= solver.z[i]

    # update convergence criteria (one for each constraint)
    solver.rᵏ[i] = norm(solver.regTrafo[i] * solver.x - solver.z[i])  # primal residual (x-z)
    solver.sᵏ[i] = norm(solver.ρ[i] * adjoint(solver.regTrafo[i]) * (solver.z[i] .- solver.zᵒˡᵈ[i])) # dual residual (concerning f(x))

    solver.ɛᵖʳⁱ[i] = max(norm(solver.regTrafo[i] * solver.x), norm(solver.z[i]))
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
      println("rᵏ[$i]/ɛᵖʳⁱ[$i] = $(solver.rᵏ[i]/solver.ɛᵖʳⁱ[i])")
      println("sᵏ[$i]/ɛᵈᵘᵃ[$i] = $(solver.sᵏ[i]/solver.ɛᵈᵘᵃ[i])")
      println("Δ[$i]/Δᵒˡᵈ[$i]  = $(solver.Δ[i]/Δᵒˡᵈ)")
      println("new ρ[$i]      = $(solver.ρ[i])")
      flush(stdout)
    end
  end

  return solver.rᵏ, iteration+1
end

function converged(solver::ADMM)
  for i ∈ eachindex(solver.reg)
    (solver.rᵏ[i] >= solver.σᵃᵇˢ + solver.relTol * solver.ɛᵖʳⁱ[i]) && return false
    (solver.sᵏ[i] >= solver.σᵃᵇˢ + solver.relTol * solver.ɛᵈᵘᵃ[i]) && return false
  end
  return true
end

@inline done(solver::ADMM,iteration::Int) = converged(solver) || iteration >= solver.iterations