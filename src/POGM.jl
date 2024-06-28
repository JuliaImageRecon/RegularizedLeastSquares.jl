export pogm, POGM

mutable struct POGM{rT<:Real,vecT<:Union{AbstractVector{rT},AbstractVector{Complex{rT}}},matA,matAHA,R,RN} <: AbstractProximalGradientSolver
  A::matA
  AHA::matAHA
  reg::R
  proj::Vector{RN}
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
  iterations::Int64
  relTol::rT
  normalizeReg::AbstractRegularizationNormalization
  norm_x₀::rT
  rel_res_norm::rT
  verbose::Bool
  restart::Symbol
end

"""
    POGM(A; AHA = A'*A, reg = L1Regularization(zero(real(eltype(AHA)))), normalizeReg = NoNormalization(), iterations = 50, verbose = false, rho = 0.95 / power_iterations(AHA), theta = 1, sigma_fac = 1, relTol = eps(real(eltype(AHA))), restart = :none)
    POGM( ; AHA = ,     reg = L1Regularization(zero(real(eltype(AHA)))), normalizeReg = NoNormalization(), iterations = 50, verbose = false, rho = 0.95 / power_iterations(AHA), theta = 1, sigma_fac = 1, relTol = eps(real(eltype(AHA))), restart = :none)

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
  * `rho::Real`                                         - step size for gradient step; the default is `0.95 / max_eigenvalue` as determined with power iterations.
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
            , iterations = 50
            , verbose = false
            , rho = 0.95 / power_iterations(AHA; verbose)
            , theta = 1
            , sigma_fac = 1
            , relTol = eps(real(eltype(AHA)))
            , restart = :none
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

  reg = isa(reg, AbstractVector) ? reg : [reg]
  indices = findsinks(AbstractProjectionRegularization, reg)
  other = [reg[i] for i in indices]
  deleteat!(reg, indices)
  if length(reg) != 1
    error("POGM does not allow for more additional regularization terms, found $(length(reg))")
  end
  other = identity.(other)
  reg = normalize(POGM, normalizeReg, reg, A, nothing)

  return POGM(A, AHA, reg[1], other, x, x₀, xᵒˡᵈ, y, z, w, res, rT(rho), rT(theta), rT(theta), rT(0), rT(1), rT(1), rT(1), rT(1), rT(sigma_fac),
    iterations, rT(relTol), normalizeReg, one(rT), rT(Inf), verbose, restart)
end

"""
    init!(it::POGM, b::vecT, x::vecT=similar(b,0), theta::Number=1)

(re-) initializes the POGM iterator
"""
function init!(solver::POGM, b; x0=0, theta=1)
  if solver.A === nothing
    solver.x₀ .= b
  else
    mul!(solver.x₀, adjoint(solver.A), b)
  end

  solver.norm_x₀ = norm(solver.x₀)

  solver.x .= x0
  solver.xᵒˡᵈ .= 0 # makes no difference in 1st iteration what this is set to
  solver.y .= 0
  solver.z .= 0
  if solver.restart != :none #save time if not using restart
    solver.w .= 0
  end

  solver.theta = theta
  solver.thetaᵒˡᵈ = theta
  solver.σ = 1
  # normalization of regularization parameters
  solver.reg = normalize(solver, solver.normalizeReg, solver.reg, solver.A, solver.x₀)
end

solverconvergence(solver::POGM) = (; :residual => norm(solver.res))

"""
  iterate(it::POGM, iteration::Int=0)

performs one POGM iteration.
"""
function iterate(solver::POGM, iteration::Int=0)
  if done(solver, iteration)
    return nothing
  end

  # calculate residuum and do gradient step
  # solver.x .-= solver.ρ .* (solver.AHA * solver.x .- solver.x₀)
  solver.xᵒˡᵈ .= solver.x #save this for inertia step later
  mul!(solver.res, solver.AHA, solver.x)
  solver.res .-= solver.x₀
  solver.x .-= solver.ρ .* solver.res

  solver.rel_res_norm = norm(solver.res) / solver.norm_x₀
  solver.verbose && println("Iteration $iteration; rel. residual = $(solver.rel_res_norm)")

  # inertial parameters
  solver.thetaᵒˡᵈ = solver.theta
  if iteration == solver.iterations - 1 && solver.restart != :none #the convergence rate depends on choice of # iterations!
    solver.theta = (1 + sqrt(1 + 8 * solver.thetaᵒˡᵈ^2)) / 2
  else
    solver.theta = (1 + sqrt(1 + 4 * solver.thetaᵒˡᵈ^2)) / 2
  end
  solver.α = (solver.thetaᵒˡᵈ - 1) / solver.theta
  solver.β = solver.σ * solver.thetaᵒˡᵈ / solver.theta
  solver.γᵒˡᵈ = solver.γ
  if solver.restart == :gradient
    solver.γ = solver.ρ * (1 + solver.α + solver.β)
  else
    solver.γ = solver.ρ * (2solver.thetaᵒˡᵈ + solver.theta - 1) / solver.theta
  end

  # inertia steps
  # x + α * (x - y) + β * (x - xᵒˡᵈ) + ρα/γᵒˡᵈ * (z - xᵒˡᵈ)
  tmp = solver.y
  solver.y = solver.x
  solver.x = tmp # swap x and y
  solver.x .*= -solver.α # here we calculate -α * y, where y is now stored in x
  solver.x .+= (1 + solver.α + solver.β) .* solver.y
  solver.x .-= (solver.β + solver.ρ * solver.α / solver.γᵒˡᵈ) .* solver.xᵒˡᵈ
  solver.x .+= solver.ρ * solver.α / solver.γᵒˡᵈ .* solver.z
  solver.z .= solver.x #store this for next iteration and GR

  # proximal map
  prox!(solver.reg, solver.x, solver.γ * λ(solver.reg))
  for proj in solver.proj
    prox!(proj, solver.x)
  end

  # gradient restart conditions
  if solver.restart == :gradient
    solver.w .+= solver.y .+ solver.ρ ./ solver.γ .* (solver.x .- solver.z)
    if real((solver.w ⋅ solver.x - solver.w ⋅ solver.z) / solver.γ - solver.w ⋅ solver.res) < 0
      solver.verbose && println("Gradient restart at iter $iteration")
      solver.σ = 1
      solver.theta = 1
    else # decreasing γ
      solver.σ *= solver.σ_fac
    end
    solver.w .= solver.ρ / solver.γ .* (solver.z .- solver.x) .- solver.y
  end

  # return the residual-norm as item and iteration number as state
  return solver, iteration + 1
end

@inline converged(solver::POGM) = (solver.rel_res_norm < solver.relTol)

@inline done(solver::POGM, iteration) = converged(solver) || iteration >= solver.iterations
