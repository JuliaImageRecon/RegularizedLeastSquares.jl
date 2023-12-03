export optista, OptISTA

mutable struct OptISTA{rT <: Real, vecT <: Union{AbstractVector{rT}, AbstractVector{Complex{rT}}}, matA, matAHA, R, RN} <: AbstractProximalGradientSolver
  A::matA
  AHA::matAHA
  reg::R
  proj::Vector{RN}
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
  iterations::Int64
  relTol::rT
  normalizeReg::AbstractRegularizationNormalization
  norm_x₀::rT
  rel_res_norm::rT
  verbose::Bool
end

"""
    OptISTA(A; AHA=A'*A, reg=L1Regularization(zero(eltype(AHA))), normalizeReg=NoNormalization(), rho=0.95, normalize_rho=true, theta=1, relTol=eps(real(eltype(AHA))), iterations=50, verbose = false)
    OptISTA( ; AHA=,     reg=L1Regularization(zero(eltype(AHA))), normalizeReg=NoNormalization(), rho=0.95, normalize_rho=true, theta=1, relTol=eps(real(eltype(AHA))), iterations=50, verbose = false)

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

See also [`createLinearSolver`](@ref), [`solve`](@ref).
"""
OptISTA(; AHA, reg = L1Regularization(zero(eltype(AHA))), normalizeReg = NoNormalization(), rho = 0.95, normalize_rho = true, theta = 1, relTol = eps(real(eltype(AHA))), iterations = 50, verbose = false) = OptISTA(nothing; AHA, reg, normalizeReg, rho, normalize_rho, theta, relTol, iterations, verbose)

function OptISTA(A
               ; AHA = A'*A
               , reg = L1Regularization(zero(eltype(AHA)))
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
  reg = vec(reg)
  indices = findsinks(AbstractProjectionRegularization, reg)
  other = [reg[i] for i in indices]
  deleteat!(reg, indices)
  if length(reg) != 1
    error("OptISTA does not allow for more additional regularization terms, found $(length(reg))")
  end
  other = identity.(other)
  reg = normalize(OptISTA, normalizeReg, reg, A, nothing)

  return OptISTA(A, AHA, reg[1], other, x, x₀, y, z, zᵒˡᵈ, res, rT(rho),rT(theta),rT(theta),rT(θn),rT(0),rT(1),rT(1),
    iterations,rT(relTol),normalizeReg,one(rT),rT(Inf),verbose)
end

"""
    init!(it::OptISTA, b::vecT
              ; A=solver.A
              , x::vecT=similar(b,0)
              , t::Float64=1.0) where T

(re-) initializes the OptISTA iterator
"""
function init!(solver::OptISTA{rT,vecT,matA,matAHA}, b::vecT
              ; x::vecT=similar(b,0)
              , θ=1
              ) where {rT,vecT,matA,matAHA}


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
  solver.y .= solver.x
  solver.z .= solver.x
  solver.zᵒˡᵈ .= solver.x

  solver.θ = θ
  solver.θᵒˡᵈ = θ
  solver.θn = θ
  for _ = 1:(solver.iterations-1)
    solver.θn = (1 + sqrt(1 + 4 * solver.θn^2)) / 2
  end
  solver.θn = (1 + sqrt(1 + 8 * solver.θn^2)) / 2

  # normalization of regularization parameters
  solver.reg = normalize(solver, solver.normalizeReg, solver.reg, solver.A, solver.x₀)
end

"""
    solve(solver::OptISTA, b::Vector)

solves an inverse problem using OptISTA.

# Arguments
* `solver::OptISTA`                 - the solver containing both system matrix and regularizer
* `b::vecT`                         - data vector

# Keywords
* `startVector::vecT=similar(b,0)`  - initial guess for the solution
* `solverInfo=nothing`              - solverInfo object

when a `SolverInfo` objects is passed, the residuals are stored in `solverInfo.convMeas`.
"""
function solve(solver::OptISTA, b; A=solver.A, startVector=similar(b,0), solverInfo=nothing, kargs...)
  # initialize solver parameters
  init!(solver, b; x=startVector)

  # log solver information
  solverInfo !== nothing && storeInfo(solverInfo,solver.x,norm(solver.res))

  # perform OptISTA iterations
  for (iteration, item) = enumerate(solver)
    solverInfo !== nothing && storeInfo(solverInfo,solver.x,norm(solver.res))
  end

  return solver.x
end

"""
  iterate(it::OptISTA, iteration::Int=0)

performs one OptISTA iteration.
"""
function iterate(solver::OptISTA, iteration::Int=0)
  if done(solver, iteration) return nothing end

  # inertial parameters
  solver.γ = 2solver.θ / solver.θn^2 * (solver.θn^2 - 2solver.θ^2 + solver.θ)
  solver.θᵒˡᵈ = solver.θ
  if iteration == solver.iterations - 1 #the convergence rate depends on choice of # iterations!
    solver.θ = (1 + sqrt(1 + 8 * solver.θᵒˡᵈ^2)) / 2
  else
    solver.θ = (1 + sqrt(1 + 4 * solver.θᵒˡᵈ^2)) / 2
  end
  solver.α = (solver.θᵒˡᵈ - 1) / solver.θ
  solver.β = solver.θᵒˡᵈ / solver.θ

  # calculate residuum and do gradient step
  # solver.y .-= solver.ρ * solver.γ .* (solver.AHA * solver.x .- solver.x₀)
  solver.zᵒˡᵈ .= solver.z #store this for inertia step
  solver.z .= solver.y #save yᵒˡᵈ in the variable z
  mul!(solver.res, solver.AHA, solver.x)
  solver.res .-= solver.x₀
  solver.y .-= solver.ρ * solver.γ .* solver.res

  solver.rel_res_norm = norm(solver.res) / solver.norm_x₀
  solver.verbose && println("Iteration $iteration; rel. residual = $(solver.rel_res_norm)")

  # proximal map
  prox!(solver.reg, solver.y, solver.ρ * solver.γ * λ(solver.reg))

  # inertia steps
  # z = x + (y - yᵒˡᵈ) / γ
  # x = z + α * (z - zᵒˡᵈ) + β * (z - x)
  solver.z ./= -solver.γ #yᵒˡᵈ is already stored in z
  solver.z .+= solver.x .+ solver.y ./ solver.γ
  solver.x .*= -solver.β
  solver.x .+= (1 + solver.α + solver.β) .* solver.z
  solver.x .-= solver.α .* solver.zᵒˡᵈ

  # return the residual-norm as item and iteration number as state
  return solver, iteration+1
end

@inline converged(solver::OptISTA) = (solver.rel_res_norm < solver.relTol)

@inline done(solver::OptISTA,iteration) = converged(solver) || iteration>=solver.iterations
