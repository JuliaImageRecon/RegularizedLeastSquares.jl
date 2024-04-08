export optista, OptISTA

mutable struct OptISTA{rT <: Real, vecT <: Union{AbstractVector{rT}, AbstractVector{Complex{rT}}}, matA, matAHA, R, RN} <: AbstractProximalGradientSolver
  A::matA
  AHA::matAHA
  reg::R
  proj::Vector{RN}
  normalizeReg::AbstractRegularizationNormalization
  verbose::Bool
  iterations::Int64
  state::AbstractSolverState{<:OptISTA}
end

mutable struct OptISTAState{rT <: Real, vecT <: Union{AbstractVector{rT}, AbstractVector{Complex{rT}}}} <: AbstractSolverState{OptISTA}
  x::vecT
  x₀::vecT
  y::vecT
  z::vecT
  zᵒˡᵈ::vecT
  res::vecT
  ρ::rT
  θ::rT
  θᵒˡᵈ::rT
  θn::rT
  α::rT
  β::rT
  γ::rT
  iteration::Int64
  relTol::rT
  norm_x₀::rT
  rel_res_norm::rT
end

"""
    OptISTA(A; AHA=A'*A, reg=L1Regularization(zero(real(eltype(AHA)))), normalizeReg=NoNormalization(), rho=0.95, normalize_rho=true, theta=1, relTol=eps(real(eltype(AHA))), iterations=50, verbose = false)
    OptISTA( ; AHA=,     reg=L1Regularization(zero(real(eltype(AHA)))), normalizeReg=NoNormalization(), rho=0.95, normalize_rho=true, theta=1, relTol=eps(real(eltype(AHA))), iterations=50, verbose = false)

creates a `OptISTA` object for the forward operator `A` or normal operator `AHA`. OptISTA has a 2x better worst-case bound than FISTA, but actual performance varies by application. It stores 2 extra intermediate variables the size of the image compared to FISTA.

Reference:
- Uijeong Jang, Shuvomoy Das Gupta, Ernest K. Ryu, "Computer-Assisted Design of Accelerated Composite Optimization Methods: OptISTA," arXiv:2305.15704, 2023, [https://arxiv.org/abs/2305.15704]

# Required Arguments
* `A`                                                 - forward operator
OR
* `AHA`                                               - normal operator (as a keyword argument)

# Optional Keyword Arguments
* `AHA`                                               - normal operator is optional if `A` is supplied
* `reg::AbstractParameterizedRegularization`          - regularization term
* `normalizeReg::AbstractRegularizationNormalization` - regularization normalization scheme; options are `NoNormalization()`, `MeasurementBasedNormalization()`, `SystemMatrixBasedNormalization()`
* `rho::Real`                                         - step size for gradient step
* `normalize_rho::Bool`                               - normalize step size by the largest eigenvalue of `AHA`
* `theta::Real`                                       - parameter for predictor-corrector step
* `relTol::Real`                                      - tolerance for stopping criterion
* `iterations::Int`                                   - maximum number of iterations
* `verbose::Bool`                                     - print residual in each iteration

See also [`createLinearSolver`](@ref), [`solve!`](@ref).
"""
OptISTA(; AHA, kwargs...) = OptISTA(nothing; AHA = AHA, kwargs...)

function OptISTA(A
               ; AHA = A'*A
               , reg = L1Regularization(zero(real(eltype(AHA))))
               , normalizeReg = NoNormalization()
               , rho = 0.95
               , normalize_rho = true
               , theta = 1
               , relTol = eps(real(eltype(AHA)))
               , iterations = 50
               , verbose = false
               )

  T  = eltype(AHA)
  rT = real(T)

  x    = Vector{T}(undef,size(AHA,2))
  x₀   = similar(x)
  y    = similar(x)
  z    = similar(x)
  zᵒˡᵈ = similar(x)
  res  = similar(x)
  res[1] = Inf # avoid spurious convergence in first iterations

  if normalize_rho
    rho /= abs(power_iterations(AHA))
  end
  θn = 1
  for _ = 1:(iterations-1)
    θn = (1 + sqrt(1 + 4 * θn^2)) / 2
  end
  θn = (1 + sqrt(1 + 8 * θn^2)) / 2

  # Prepare regularization terms
  reg = isa(reg, AbstractVector) ? reg : [reg]
  indices = findsinks(AbstractProjectionRegularization, reg)
  other = [reg[i] for i in indices]
  deleteat!(reg, indices)
  if length(reg) != 1
    error("OptISTA does not allow for more additional regularization terms, found $(length(reg))")
  end
  other = identity.(other)
  reg = normalize(OptISTA, normalizeReg, reg, A, nothing)

  state = OptISTAState(x, x₀, y, z, zᵒˡᵈ, res, rT(rho),rT(theta),rT(theta),rT(θn),rT(0),rT(1),rT(1),
  0,rT(relTol), one(rT),rT(Inf))

  return OptISTA(A, AHA, reg[1], other, normalizeReg, verbose, iterations, state)
end

function init!(solver::OptISTA, state::OptISTAState, b; kwargs...)
  x = similar(b, size(state.x)...)
  x₀ = similar(b, size(state.x₀)...)
  y = similar(b, size(state.y)...)
  z = similar(b, size(state.z)...)
  zᵒˡᵈ = similar(b, size(state.zᵒˡᵈ)...)
  res = similar(b, size(state.res)...)

  state = OptISTAState(x, x₀, y, z, zᵒˡᵈ, res, state.ρ, state.θ, state.θᵒˡᵈ, state.θn, state.α, state.β, state.γ, state.iteration, state.relTol, state.norm_x₀, state.rel_res_norm)
  solver.state = state
  init!(solver, state, b; kwargs...)
end

"""
    init!(it::OptISTA, b::vecT
              ; A=solver.A
              , x::vecT=similar(b,0)
              , t::Float64=1.0) where T

(re-) initializes the OptISTA iterator
"""
function init!(solver::OptISTA, state::OptISTAState{rT, vecT}, b::vecT; x0 = 0, θ=1) where {rT, vecT}
  if solver.A === nothing
    state.x₀ .= b
  else
    mul!(state.x₀, adjoint(solver.A), b)
  end

  state.norm_x₀ = norm(state.x₀)

  state.x .= x0
  state.y .= state.x
  state.z .= state.x
  state.zᵒˡᵈ .= state.x

  state.res[:] .= rT(Inf)
  state.θ = θ
  state.θᵒˡᵈ = θ
  state.θn = θ
  for _ = 1:(solver.iterations-1)
    state.θn = (1 + sqrt(1 + 4 * state.θn^2)) / 2
  end
  state.θn = (1 + sqrt(1 + 8 * state.θn^2)) / 2
  state.rel_res_norm = rT(Inf)

  state.iteration = 0
  # normalization of regularization parameters
  solver.reg = normalize(solver, solver.normalizeReg, solver.reg, solver.A, state.x₀)
end

solverconvergence(solver::OptISTA) = (; :residual => norm(solver.res))

"""
  iterate(it::OptISTA, iteration::Int=0)

performs one OptISTA iteration.
"""
function iterate(solver::OptISTA, state::OptISTAState = solver.state)
  if done(solver, state) return nothing end

  # inertial parameters
  state.γ = 2state.θ / state.θn^2 * (state.θn^2 - 2state.θ^2 + state.θ)
  state.θᵒˡᵈ = state.θ
  if state.iteration == solver.iterations - 1 #the convergence rate depends on choice of # iterations!
    state.θ = (1 + sqrt(1 + 8 * state.θᵒˡᵈ^2)) / 2
  else
    state.θ = (1 + sqrt(1 + 4 * state.θᵒˡᵈ^2)) / 2
  end
  state.α = (state.θᵒˡᵈ - 1) / state.θ
  state.β = state.θᵒˡᵈ / state.θ

  # calculate residuum and do gradient step
  # state.y .-= state.ρ * state.γ .* (solver.AHA * state.x .- state.x₀)
  state.zᵒˡᵈ .= state.z #store this for inertia step
  state.z .= state.y #save yᵒˡᵈ in the variable z
  mul!(state.res, solver.AHA, state.x)
  state.res .-= state.x₀
  state.y .-= state.ρ * state.γ .* state.res

  state.rel_res_norm = norm(state.res) / state.norm_x₀
  solver.verbose && println("Iteration $iteration; rel. residual = $(state.rel_res_norm)")

  # proximal map
  prox!(solver.reg, state.y, state.ρ * state.γ * λ(solver.reg))

  # inertia steps
  # z = x + (y - yᵒˡᵈ) / γ
  # x = z + α * (z - zᵒˡᵈ) + β * (z - x)
  state.z ./= -state.γ #yᵒˡᵈ is already stored in z
  state.z .+= state.x .+ state.y ./ state.γ
  state.x .*= -state.β
  state.x .+= (1 + state.α + state.β) .* state.z
  state.x .-= state.α .* state.zᵒˡᵈ

  state.iteration += 1
  # return the residual-norm as item and iteration number as state
  return state.x, state
end

@inline converged(solver::OptISTA, state::OptISTAState) = (state.rel_res_norm < state.relTol)

@inline done(solver::OptISTA, state::OptISTAState) = converged(solver, state) || state.iteration >= solver.iterations

solversolution(solver::OptISTA) = solver.state.x 