export optista, OptISTA

mutable struct OptISTA{rT <: Real, vecT <: Union{AbstractVector{rT}, AbstractVector{Complex{rT}}}, matA, matAHA} <: AbstractLinearSolver
  A::matA
  AᴴA::matAHA
  reg::AbstractRegularization
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
    OptISTA(A, x::vecT=zeros(eltype(A),size(A,2))
          ; reg=nothing, regName=["L1"], λ=[0.0], kargs...)

creates a `OptISTA` object for the system matrix `A`.
OptISTA has a 2x better worst-case bound than FISTA, but actual performance varies by application.
It stores 2 extra intermediate variables the size of the image compared to FISTA

Reference:
- Uijeong Jang, Shuvomoy Das Gupta, Ernest K. Ryu,
    "Computer-Assisted Design of Accelerated Composite 
    Optimization Methods: OptISTA," arXiv:2305.15704, 2023,
    [https://arxiv.org/abs/2305.15704]

# Arguments
* `A`                       - system matrix
* `x::vecT`                 - array with the same type and size as the solution
* (`reg=nothing`)           - regularization object
* (`regName=["L1"]`)        - name of the Regularization to use (if reg==nothing)
* (`AᴴA=A'*A`)              - specialized normal operator, default is `A'*A`
* (`λ=0`)                   - regularization parameter
* (`ρ=0.95`)                - step size for gradient step
* (`normalize_ρ=false`)     - normalize step size by the maximum eigenvalue of `AᴴA`
* (`θ=1.0`)                 - parameter for predictor-corrector step
* (`relTol::Float64=1.e-5`) - tolerance for stopping criterion
* (`iterations::Int64=50`)  - maximum number of iterations
"""
function OptISTA(A, x::AbstractVector{T}=Vector{eltype(A)}(undef,size(A,2)); reg=L1Regularization(zero(T))
              , AᴴA=A'*A
              , ρ=0.95
              , normalize_ρ=true
              , θ=1
              , relTol=eps(real(T))
              , iterations=50
              , normalizeReg=NoNormalization()
              , verbose = false
              , kargs...) where {T}

  rT = real(T)

  x₀ = similar(x)
  y = similar(x)
  z = similar(x)
  zᵒˡᵈ = similar(x)
  res  = similar(x)
  res[1] = Inf # avoid spurious convergence in first iterations

  if normalize_ρ
    ρ /= abs(power_iterations(AᴴA))
  end
  θn = 1
  for _ = 1:(iterations-1)
    θn = (1 + sqrt(1 + 4 * θn^2)) / 2
  end
  θn = (1 + sqrt(1 + 8 * θn^2)) / 2

  reg = normalize(OptISTA, normalizeReg, vec(reg), A, nothing)

  return OptISTA(A, AᴴA, vec(reg)[1], x, x₀, y, z, zᵒˡᵈ, res, rT(ρ),rT(θ),rT(θ),rT(θn),rT(0),rT(1),rT(1),
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

  solver.x₀ .= adjoint(solver.A) * b
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
  solver.reg = normalize(solver, solver.normalizeReg, vec(solver.reg), solver.A, solver.x₀)[1]
end

"""
    solve(solver::OptISTA, b::Vector)

solves an inverse problem using OptISTA.

# Arguments
* `solver::OptISTA`                     - the solver containing both system matrix and regularizer
* `b::vecT`                           - data vector
* `A=solver.A`                        - operator for the data-term of the problem
* (`startVector::vecT=similar(b,0)`)  - initial guess for the solution
* (`solverInfo=nothing`)              - solverInfo object

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
  # solver.y .-= solver.ρ * solver.γ .* (solver.AᴴA * solver.x .- solver.x₀)
  solver.zᵒˡᵈ .= solver.z #store this for inertia step
  solver.z .= solver.y #save yᵒˡᵈ in the variable z
  mul!(solver.res, solver.AᴴA, solver.x)
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
