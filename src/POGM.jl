export pogm, POGM

mutable struct POGM{matA,matAHA,R,RN} <: AbstractProximalGradientSolver
  A::matA
  AHA::matAHA
  reg::R
  proj::Vector{RN}
  normalizeReg::AbstractRegularizationNormalization
  verbose::Bool
  restart::Symbol  
  iterations::Int64
  state::AbstractSolverState{<:POGM}
end

mutable struct POGMState{rT <: Real, vecT <: Union{AbstractVector{rT}, AbstractVector{Complex{rT}}}} <: AbstractSolverState{POGM}
  x::vecT
  x₀::vecT
  xᵒˡᵈ::vecT
  y::vecT
  z::vecT
  w::vecT
  res::vecT
  ρ::rT
  theta::rT
  thetaᵒˡᵈ::rT
  α::rT
  β::rT
  γ::rT
  γᵒˡᵈ::rT
  σ::rT
  σ_fac::rT
  iteration::Int64
  relTol::rT
  norm_x₀::rT
  rel_res_norm::rT
end
"""
    POGM(A; AHA = A'*A, reg = L1Regularization(zero(real(eltype(AHA)))), normalizeReg = NoNormalization(), rho = 0.95, normalize_rho = true, theta = 1, sigma_fac = 1, relTol = eps(real(eltype(AHA))), iterations = 50, restart = :none, verbose = false)
    POGM( ; AHA = ,     reg = L1Regularization(zero(real(eltype(AHA)))), normalizeReg = NoNormalization(), rho = 0.95, normalize_rho = true, theta = 1, sigma_fac = 1, relTol = eps(real(eltype(AHA))), iterations = 50, restart = :none, verbose = false)

Creates a `POGM` object for the forward operator `A` or normal operator `AHA`. POGM has a 2x better worst-case bound than FISTA, but actual performance varies by application. It stores 3 extra intermediate variables the size of the image compared to FISTA. Only gradient restart scheme is implemented for now.

# References:
- A.B. Taylor, J.M. Hendrickx, F. Glineur,
    "Exact worst-case performance of first-order algorithms
    for composite convex optimization," Arxiv:1512.07516, 2015,
    SIAM J. Opt. 2017
    [http://doi.org/10.1137/16m108104x]
- Kim, D., & Fessler, J. A. (2018).
    Adaptive Restart of the Optimized Gradient Method for Convex Optimization.
    Journal of Optimization Theory and Applications, 178(1), 240–263.
    [https://doi.org/10.1007/s10957-018-1287-4]

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
  * `sigma_fac::Real`                                   - parameter for decreasing γ-momentum ∈ [0,1]
  * `relTol::Real`                                      - tolerance for stopping criterion
  * `iterations::Int`                                   - maximum number of iterations
  * `restart::Symbol`                                   - `:none`, `:gradient` options for restarting
  * `verbose::Bool`                                     - print residual in each iteration

See also [`createLinearSolver`](@ref), [`solve!`](@ref).
"""
POGM(; AHA, kwargs...) = POGM(nothing; kwargs..., AHA = AHA)

function POGM(A
            ; AHA = A'*A
            , reg = L1Regularization(zero(real(eltype(AHA))))
            , normalizeReg = NoNormalization()
            , rho = 0.95
            , normalize_rho = true
            , theta = 1
            , sigma_fac = 1
            , relTol = eps(real(eltype(AHA)))
            , iterations = 50
            , restart = :none
            , verbose = false
)

  T = eltype(AHA)
  rT = real(T)

  x = Vector{T}(undef, size(AHA, 2))
  x₀ = similar(x)
  xᵒˡᵈ = similar(x)
  y = similar(x)
  z = similar(x)
  w = similar(x)
  res = similar(x)
  res[1] = Inf # avoid spurious convergence in first iterations

  if normalize_rho
    rho /= abs(power_iterations(AHA))
  end

  reg = isa(reg, AbstractVector) ? reg : [reg]
  indices = findsinks(AbstractProjectionRegularization, reg)
  other = [reg[i] for i in indices]
  deleteat!(reg, indices)
  if length(reg) != 1
    error("POGM does not allow for more additional regularization terms, found $(length(reg))")
  end
  other = identity.(other)
  reg = normalize(POGM, normalizeReg, reg, A, nothing)

  state = POGMState(x, x₀, xᵒˡᵈ, y, z, w, res, rT(rho), rT(theta), rT(theta), rT(0), rT(1), rT(1),
   rT(1), rT(1), rT(sigma_fac), 0, rT(relTol), one(rT), rT(Inf))

  return POGM(A, AHA, reg[1], other, normalizeReg, verbose, restart, iterations, state)
end

function init!(solver::POGM, state::POGMState{rT, vecT}, b::otherT; kwargs...) where {rT, vecT, otherT}
  x = similar(b, size(state.x)...)
  x₀ = similar(b, size(state.x₀)...)
  xᵒˡᵈ = similar(b, size(state.xᵒˡᵈ)...)
  y = similar(b, size(state.y)...)
  z = similar(b, size(state.z)...)
  w = similar(b, size(state.w)...)
  res = similar(b, size(state.res)...)

  state = POGMState(x, x₀, xᵒˡᵈ, y, z, w, res, state.ρ, state.theta, state.theta,
    state.α, state.β, state.γ, state.γᵒˡᵈ, state.σ, state.σ_fac,
    state.iteration, state.relTol, state.norm_x₀, state.rel_res_norm)
  
  solver.state = state
  init!(solver, state, b; kwargs...)
end

"""
    init!(it::POGM, b::vecT, x::vecT=similar(b,0), theta::Number=1)

(re-) initializes the POGM iterator
"""
function init!(solver::POGM, state::POGMState{rT, vecT}, b::vecT; x0 = 0, theta=1) where {rT, vecT}
  if solver.A === nothing
    state.x₀ .= b
  else
    mul!(state.x₀, adjoint(solver.A), b)
  end

  state.norm_x₀ = norm(state.x₀)

  state.x .= x0
  state.xᵒˡᵈ .= 0 # makes no difference in 1st iteration what this is set to
  state.y .= 0
  state.z .= 0
  if solver.restart != :none #save time if not using restart
    state.w .= 0
  end

  state.res[:] .= rT(Inf)
  state.theta = theta
  state.thetaᵒˡᵈ = theta
  state.σ = 1
  state.rel_res_norm = rT(Inf)

  state.iteration = 0
  # normalization of regularization parameters
  solver.reg = normalize(solver, solver.normalizeReg, solver.reg, solver.A, state.x₀)
end

solverconvergence(solver::POGM) = (; :residual => norm(solver.res))

"""
  iterate(it::POGM, iteration::Int=0)

performs one POGM iteration.
"""
function iterate(solver::POGM, state = solver.state)
  if done(solver, state)
    return nothing
  end

  # calculate residuum and do gradient step
  # solver.x .-= solver.ρ .* (solver.AHA * solver.x .- solver.x₀)
  state.xᵒˡᵈ .= state.x #save this for inertia step later
  mul!(state.res, solver.AHA, state.x)
  state.res .-= state.x₀
  state.x .-= state.ρ .* state.res

  state.rel_res_norm = norm(state.res) / state.norm_x₀
  solver.verbose && println("Iteration $iteration; rel. residual = $(state.rel_res_norm)")

  # inertial parameters
  state.thetaᵒˡᵈ = state.theta
  if state.iteration == solver.iterations - 1 && solver.restart != :none #the convergence rate depends on choice of # iterations!
    state.theta = (1 + sqrt(1 + 8 * state.thetaᵒˡᵈ^2)) / 2
  else
    state.theta = (1 + sqrt(1 + 4 * state.thetaᵒˡᵈ^2)) / 2
  end
  state.α = (state.thetaᵒˡᵈ - 1) / state.theta
  state.β = state.σ * state.thetaᵒˡᵈ / state.theta
  state.γᵒˡᵈ = state.γ
  if solver.restart == :gradient
    state.γ = state.ρ * (1 + state.α + state.β)
  else
    state.γ = state.ρ * (2state.thetaᵒˡᵈ + state.theta - 1) / state.theta
  end

  # inertia steps
  # x + α * (x - y) + β * (x - xᵒˡᵈ) + ρα/γᵒˡᵈ * (z - xᵒˡᵈ)
  tmp = state.y
  state.y = state.x
  state.x = tmp # swap x and y
  state.x .*= -state.α # here we calculate -α * y, where y is now stored in x
  state.x .+= (1 + state.α + state.β) .* state.y
  state.x .-= (state.β + state.ρ * state.α / state.γᵒˡᵈ) .* state.xᵒˡᵈ
  state.x .+= state.ρ * state.α / state.γᵒˡᵈ .* state.z
  state.z .= state.x #store this for next iteration and GR

  # proximal map
  prox!(solver.reg, state.x, state.γ * λ(solver.reg))
  for proj in solver.proj
    prox!(proj, state.x)
  end

  # gradient restart conditions
  if solver.restart == :gradient
    state.w .+= state.y .+ state.ρ ./ state.γ .* (state.x .- state.z)
    if real((state.w ⋅ state.x - state.w ⋅ state.z) / state.γ - state.w ⋅ state.res) < 0
      solver.verbose && println("Gradient restart at iter $iteration")
      state.σ = 1
      state.theta = 1
    else # decreasing γ
      state.σ *= state.σ_fac
    end
    state.w .= state.ρ / state.γ .* (state.z .- state.x) .- state.y
  end

  # return the residual-norm as item and iteration number as state
  state.iteration += 1
  return state.x, state
end

@inline converged(solver::POGM, state::POGMState) = (state.rel_res_norm < state.relTol)

@inline done(solver::POGM, state::POGMState) = converged(solver, state) || state.iteration >= solver.iterations

solversolution(solver::POGM) = solver.state.x 