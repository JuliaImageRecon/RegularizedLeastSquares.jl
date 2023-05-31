export pogm, POGM

mutable struct POGM{rT <: Real, vecT <: Union{AbstractVector{rT}, AbstractVector{Complex{rT}}}, matA, matAHA} <: AbstractLinearSolver
  A::matA
  AᴴA::matAHA
  reg::Regularization
  x::vecT
  x₀::vecT
  xᵒˡᵈ::vecT
  y::vecT
  z::vecT
  w::vecT
  res::vecT
  ρ::rT
  t::rT
  tᵒˡᵈ::rT
  α::rT
  β::rT
  γ::rT
  γᵒˡᵈ::rT
  σ::rT
  σ_fac::rT
  iterations::Int64
  relTol::rT
  normalizeReg::Bool
  regFac::rT
  norm_x₀::rT
  rel_res_norm::rT
  verbose::Bool
  restart::Symbol
end

"""
    POGM(A, x::vecT=zeros(eltype(A),size(A,2))
          ; reg=nothing, regName=["L1"], λ=[0.0], kargs...)

creates a `POGM` object for the system matrix `A`.
POGM has a 2x better worst-case bound than FISTA, but actual performance varies by application.
It stores 3 extra intermediate variables the size of the image compared to FISTA
Only gradient restart scheme is implemented for now

References:
- A.B. Taylor, J.M. Hendrickx, F. Glineur,
    "Exact worst-case performance of first-order algorithms
    for composite convex optimization," Arxiv:1512.07516, 2015,
    SIAM J. Opt. 2017
    [http://doi.org/10.1137/16m108104x]
- Kim, D., & Fessler, J. A. (2018).
    Adaptive Restart of the Optimized Gradient Method for Convex Optimization.
    Journal of Optimization Theory and Applications, 178(1), 240–263.
    [https://doi.org/10.1007/s10957-018-1287-4]

# Arguments
* `A`                       - system matrix
* `x::vecT`                 - array with the same type and size as the solution
* (`reg=nothing`)           - regularization object
* (`regName=["L1"]`)        - name of the Regularization to use (if reg==nothing)
* (`AᴴA=A'*A`)              - specialized normal operator, default is `A'*A`
* (`λ=0`)                   - regularization parameter
* (`ρ=0.95`)                - step size for gradient step
* (`normalize_ρ=false`)     - normalize step size by the maximum eigenvalue of `AᴴA`
* (`t=1.0`)                 - parameter for predictor-corrector step
* (`σ_fac=1.0`)             - parameter for decreasing γ-momentum ∈ [0,1]
* (`relTol::Float64=1.e-5`) - tolerance for stopping criterion
* (`iterations::Int64=50`)  - maximum number of iterations
* (`restart::Symbol=:none`) - :none, :gradient options for restarting
"""
function POGM(A, x::AbstractVector{T}=Vector{eltype(A)}(undef,size(A,2)); reg=nothing, regName=["L1"]
              , AᴴA=A'*A
              , λ=0
              , ρ=0.95
              , normalize_ρ=true
              , t=1
              , σ_fac=1.0
              , relTol=eps(real(T))
              , iterations=50
              , normalizeReg=false
              , restart = :none
              , verbose = false
              , kargs...) where {T}

  rT = real(T)
  if reg == nothing
    reg = Regularization(regName, λ, kargs...)
  end

  x₀ = similar(x)
  xᵒˡᵈ = similar(x)
  y = similar(x)
  z = similar(x)
  w = similar(x)
  res  = similar(x)
  res[1] = Inf # avoid spurious convergence in first iterations

  if normalize_ρ
    ρ /= abs(power_iterations(AᴴA))
  end

  return POGM(A, AᴴA, vec(reg)[1], x, x₀, xᵒˡᵈ, y, z, w, res, rT(ρ),rT(t),rT(t),rT(0),rT(1),rT(1),rT(1),rT(1),rT(σ_fac),
    iterations,rT(relTol),normalizeReg,one(rT),one(rT),rT(Inf),verbose,restart)
end

"""
    init!(it::POGM, b::vecT
              ; A=solver.A
              , x::vecT=similar(b,0)
              , t::Float64=1.0) where T

(re-) initializes the POGM iterator
"""
function init!(solver::POGM{rT,vecT,matA,matAHA}, b::vecT
              ; x::vecT=similar(b,0)
              , t=1
              ) where {rT,vecT,matA,matAHA}

  solver.x₀ .= adjoint(solver.A) * b
  solver.norm_x₀ = norm(solver.x₀)

  if isempty(x)
    solver.x .= 0
  else
    solver.x .= x
  end
  solver.xᵒˡᵈ .= 0 # makes no difference in 1st iteration what this is set to
  solver.y .= 0
  solver.z .= 0
  if solver.restart != :none #save time if not using restart
    solver.w .= 0
  end

  solver.t = t
  solver.tᵒˡᵈ = t
  solver.γ = 1 #doesn't matter
  solver.γᵒˡᵈ = 1 #doesn't matter
  solver.σ = 1
  # normalization of regularization parameters
  if solver.normalizeReg
    solver.regFac = norm(solver.x₀,1)/length(solver.x₀)
  else
    solver.regFac = 1
  end
end

"""
    solve(solver::POGM, b::Vector)

solves an inverse problem using POGM.

# Arguments
* `solver::POGM`                     - the solver containing both system matrix and regularizer
* `b::vecT`                           - data vector
* `A=solver.A`                        - operator for the data-term of the problem
* (`startVector::vecT=similar(b,0)`)  - initial guess for the solution
* (`solverInfo=nothing`)              - solverInfo object

when a `SolverInfo` objects is passed, the residuals are stored in `solverInfo.convMeas`.
"""
function solve(solver::POGM, b; A=solver.A, startVector=similar(b,0), solverInfo=nothing, kargs...)
  # initialize solver parameters
  init!(solver, b; x=startVector)

  # log solver information
  solverInfo != nothing && storeInfo(solverInfo,solver.x,norm(solver.res))

  # perform POGM iterations
  for (iteration, item) = enumerate(solver)
    solverInfo != nothing && storeInfo(solverInfo,solver.x,norm(solver.res))
  end

  return solver.x
end

"""
  iterate(it::POGM, iteration::Int=0)

performs one POGM iteration.
"""
function iterate(solver::POGM, iteration::Int=0)
  if done(solver, iteration) return nothing end

  # calculate residuum and do gradient step
  # solver.x .-= solver.ρ .* (solver.AᴴA * solver.x .- solver.x₀)
  solver.xᵒˡᵈ .= solver.x #save this for inertia step later
  mul!(solver.res, solver.AᴴA, solver.x)
  solver.res .-= solver.x₀
  solver.x .-= solver.ρ .* solver.res

  solver.rel_res_norm = norm(solver.res) / solver.norm_x₀
  solver.verbose && println("Iteration $iteration; rel. residual = $(solver.rel_res_norm)")

  # inertial parameters
  solver.tᵒˡᵈ = solver.t
  if iteration == solver.iterations - 1 && solver.restart != :none #the convergence rate depends on choice of # iterations!
    solver.t = (1 + sqrt(1 + 8 * solver.tᵒˡᵈ^2)) / 2
  else
    solver.t = (1 + sqrt(1 + 4 * solver.tᵒˡᵈ^2)) / 2
  end
  solver.α = (solver.tᵒˡᵈ - 1) / solver.t
  solver.β = solver.σ * solver.tᵒˡᵈ / solver.t
  solver.γᵒˡᵈ = solver.γ
  if solver.restart == :gradient
    solver.γ = solver.ρ * (1 + solver.α + solver.β)
  else
    solver.γ = solver.ρ * (2solver.tᵒˡᵈ + solver.t - 1) / solver.t
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
  solver.reg.prox!(solver.x, solver.regFac*solver.reg.λ*solver.γ; solver.reg.params...)

  # gradient restart conditions
  if solver.restart == :gradient
    solver.w .+= solver.y .+ solver.ρ ./ solver.γ .* (solver.x .- solver.z)
    if real((solver.w ⋅ solver.x - solver.w ⋅ solver.z) / solver.γ - solver.w ⋅ solver.res) < 0
      solver.verbose && println("Gradient restart at iter $iteration")
      solver.σ = 1
      solver.t = 1
    else # decreasing γ
      solver.σ *= solver.σ_fac
    end
    solver.w .= solver.ρ / solver.γ .* (solver.z .- solver.x) .- solver.y
  end

  # return the residual-norm as item and iteration number as state
  return solver, iteration+1
end

@inline converged(solver::POGM) = (solver.rel_res_norm < solver.relTol)

@inline done(solver::POGM,iteration) = converged(solver) || iteration>=solver.iterations
