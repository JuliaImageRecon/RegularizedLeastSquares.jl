export FISTA

mutable struct FISTA{rT <: Real, vecT <: Union{AbstractVector{rT}, AbstractVector{Complex{rT}}}, matA, matAHA, R, RN} <: AbstractProximalGradientSolver
  A::matA
  AHA::matAHA
  reg::R
  proj::Vector{RN}
  x::vecT
  x₀::vecT
  xᵒˡᵈ::vecT
  res::vecT
  ρ::rT
  theta::rT
  thetaᵒˡᵈ::rT
  iterations::Int64
  relTol::rT
  normalizeReg::AbstractRegularizationNormalization
  norm_x₀::rT
  rel_res_norm::rT
  verbose::Bool
  restart::Symbol
end

"""
    FISTA(A; AHA=A'*A, reg=L1Regularization(zero(eltype(AHA))), normalizeReg=NoNormalization(), rho=0.95, normalize_rho=true, theta=1, relTol=eps(real(eltype(AHA))), iterations=50, restart = :none, verbose = false)
    FISTA( ; AHA=,     reg=L1Regularization(zero(eltype(AHA))), normalizeReg=NoNormalization(), rho=0.95, normalize_rho=true, theta=1, relTol=eps(real(eltype(AHA))), iterations=50, restart = :none, verbose = false)

creates a `FISTA` object for the forward operator `A` or normal operator `AHA`.

# Required Arguments
  * `A`                                                 - forward operator
  OR
  * `AHA`                                               - normal operator (as a keyword argument)

# Optional Keyword Arguments
* `AHA`                                               - normal operator is optional if `A` is supplied
* `precon`                                            - preconditionner for the internal CG algorithm
* `reg::AbstractParameterizedRegularization`          - regularization term; can also be a vector of regularization terms
* `normalizeReg::AbstractRegularizationNormalization` - regularization normalization scheme; options are `NoNormalization()`, `MeasurementBasedNormalization()`, `SystemMatrixBasedNormalization()`
* `rho::Real`                                         - step size for gradient step
* `normalize_rho::Bool`                               - normalize step size by the largest eigenvalue of `AHA`
* `theta::Real`                                       - parameter for predictor-corrector step
* `relTol::Real`                                      - tolerance for stopping criterion
* `iterations::Int`                                   - maximum number of iterations
* `restart::Symbol`                                   - `:none`, `:gradient` options for restarting
* `verbose::Bool`                                     - print residual in each iteration

See also [`createLinearSolver`](@ref), [`solve`](@ref).
"""
FISTA(; AHA = A'*A, reg = L1Regularization(zero(eltype(AHA))), normalizeReg = NoNormalization(), rho = 0.95, normalize_rho = true, theta = 1, relTol = eps(real(eltype(AHA))), iterations = 50, restart = :none, verbose = false) = FISTA(nothing; AHA, reg, normalizeReg, rho, normalize_rho, theta, relTol, iterations, restart, verbose)

function FISTA(A
             ; AHA = A'*A
             , reg = L1Regularization(zero(eltype(AHA)))
             , normalizeReg = NoNormalization()
             , rho = 0.95
             , normalize_rho = true
             , theta = 1
             , relTol = eps(real(eltype(AHA)))
             , iterations = 50
             , restart = :none
             , verbose = false
             )

  T  = eltype(AHA)
  rT = real(T)

  x    = Vector{T}(undef,size(AHA,2))
  x₀   = similar(x)
  xᵒˡᵈ = similar(x)
  res  = similar(x)
  res[1] = Inf # avoid spurious convergence in first iterations

  if normalize_rho
    rho /= abs(power_iterations(AHA))
  end

  # Prepare regularization terms
  reg = vec(reg)
  indices = findsinks(AbstractProjectionRegularization, reg)
  other = [reg[i] for i in indices]
  deleteat!(reg, indices)
  if length(reg) != 1
    error("FISTA does not allow for more additional regularization terms, found $(length(reg))")
  end
  other = identity.(other)
  reg = normalize(FISTA, normalizeReg, reg, A, nothing)


  return FISTA(A, AHA, reg[1], other, x, x₀, xᵒˡᵈ, res, rT(rho),rT(theta),rT(theta),iterations,rT(relTol),normalizeReg,one(rT),rT(Inf),verbose,restart)
end

"""
    init!(it::FISTA, b, x=similar(b,0), theta::Number=1)

(re-) initializes the FISTA iterator
"""
function init!(solver::FISTA{rT,vecT,matA,matAHA}, b::vecT; x::vecT=similar(b,0), theta=1) where {rT,vecT,matA,matAHA}

  if solver.A === nothing
    solver.x₀ .= b
  else
    mul!(solver.x₀, adjoint(solver.A), b)
  end

  solver.norm_x₀ = norm(solver.x₀)

  if isempty(x)
    solver.x .= 0
  else
    solver.x .= x
  end
  solver.xᵒˡᵈ .= 0 # makes no difference in 1st iteration what this is set to

  solver.theta = theta
  solver.thetaᵒˡᵈ = theta
  # normalization of regularization parameters
  solver.reg = normalize(solver, solver.normalizeReg, solver.reg, solver.A, solver.x₀)
end

"""
    solve(solver::FISTA, b; startVector=similar(b,0), solverInfo=nothing)

solves an inverse problem using FISTA.

# Arguments
* `solver::FISTA`                   - the solver containing both system matrix and regularizer
* `b::AbstractVector`               - data vector if `A` was supplied to the solver, back-projection of the data otherwise

# Keyword Arguments
* `startVector::AbstractVector`     - initial guess for the solution
* `solverInfo::SolverInfo`          - solverInfo object

when a `SolverInfo` object is passed, the residuals are stored in `solverInfo.convMeas`.
"""
function solve(solver::FISTA, b; startVector=similar(b,0), solverInfo=nothing)
  # initialize solver parameters
  init!(solver, b; x=startVector)

  # log solver information
  solverInfo !== nothing && storeInfo(solverInfo,solver.x,norm(solver.res))

  # perform FISTA iterations
  for (iteration, item) = enumerate(solver)
    solverInfo !== nothing && storeInfo(solverInfo,solver.x,norm(solver.res))
  end

  return solver.x
end


"""
  iterate(it::FISTA, iteration::Int=0)

performs one fista iteration.
"""
function iterate(solver::FISTA, iteration::Int=0)
  if done(solver, iteration) return nothing end

  # momentum / Nesterov step
  # this implementation mimics BART, saving memory by first swapping x and xᵒˡᵈ before calculating x + α * (x - xᵒˡᵈ)
  tmp = solver.xᵒˡᵈ
  solver.xᵒˡᵈ = solver.x
  solver.x = tmp # swap x and xᵒˡᵈ
  solver.x .*= ((1 - solver.thetaᵒˡᵈ)/solver.theta) # here we calculate -α * xᵒˡᵈ, where xᵒˡᵈ is now stored in x
  solver.x .+= ((solver.thetaᵒˡᵈ-1)/solver.theta + 1) .* (solver.xᵒˡᵈ) # add (α+1)*x, where x is now stored in xᵒˡᵈ

  # calculate residuum and do gradient step
  # solver.x .-= solver.ρ .* (solver.AHA * solver.x .- solver.x₀)
  mul!(solver.res, solver.AHA, solver.x)
  solver.res .-= solver.x₀
  solver.x .-= solver.ρ .* solver.res

  solver.rel_res_norm = norm(solver.res) / solver.norm_x₀
  solver.verbose && println("Iteration $iteration; rel. residual = $(solver.rel_res_norm)")

  # the two lines below are equivalent to the ones above and non-allocating, but require the 5-argument mul! function to implemented for AHA, i.e. if AHA is LinearOperator, it requires LinearOperators.jl v2
  # mul!(solver.x, solver.AHA, solver.xᵒˡᵈ, -solver.ρ, 1)
  # solver.x .+= solver.ρ .* solver.x₀

  # proximal map
  prox!(solver.reg, solver.x, solver.ρ * λ(solver.reg))

  for proj in solver.proj
    prox!(proj, solver.x)
  end

  # gradient restart conditions
  if solver.restart == :gradient
    if real(solver.res ⋅ (solver.x .- solver.xᵒˡᵈ) ) > 0 #if momentum is at an obtuse angle to the negative gradient
      solver.verbose && println("Gradient restart at iter $iteration")
      solver.theta = 1
    end
  end

  # predictor-corrector update
  solver.thetaᵒˡᵈ = solver.theta
  solver.theta = (1 + sqrt(1 + 4 * solver.thetaᵒˡᵈ^2)) / 2

  # return the residual-norm as item and iteration number as state
  return solver, iteration+1
end

@inline converged(solver::FISTA) = (solver.rel_res_norm < solver.relTol)

@inline done(solver::FISTA,iteration) = converged(solver) || iteration>=solver.iterations
