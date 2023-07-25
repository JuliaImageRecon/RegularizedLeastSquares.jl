export fista, FISTA

mutable struct FISTA{rT <: Real, vecT <: Union{AbstractVector{rT}, AbstractVector{Complex{rT}}}, matA, matAHA, R<:AbstractRegularization} <: AbstractLinearSolver
  A::matA
  AᴴA::matAHA
  reg::R
  x::vecT
  x₀::vecT
  xᵒˡᵈ::vecT
  res::vecT
  ρ::rT
  t::rT
  tᵒˡᵈ::rT
  iterations::Int64
  relTol::rT
  normalizeReg::AbstractRegularizationNormalization
  regFac::rT
  norm_x₀::rT
  rel_res_norm::rT
  verbose::Bool
  restart::Symbol
end

"""
    FISTA(A, x::vecT=zeros(eltype(A),size(A,2))
          ; reg=nothing, regName=["L1"], λ=[0.0], kargs...)

creates a `FISTA` object for the system matrix `A`.

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
* (`relTol::Float64=1.e-5`) - tolerance for stopping criterion
* (`iterations::Int64=50`)  - maximum number of iterations
* (`restart::Symbol=:none`) - :none, :gradient options for restarting
"""
function FISTA(A, x::AbstractVector{T}=Vector{eltype(A)}(undef,size(A,2)); reg=L1Regularization(0)
              , AᴴA=A'*A
              , λ=0
              , ρ=0.95
              , normalize_ρ=true
              , t=1
              , relTol=eps(real(T))
              , iterations=50
              , normalizeReg=NoNormalization()
              , restart = :none
              , verbose = false
              , kargs...) where {T}

  rT = real(T)

  x₀   = similar(x)
  xᵒˡᵈ = similar(x)
  res  = similar(x)
  res[1] = Inf # avoid spurious convergence in first iterations

  if normalize_ρ
    ρ /= abs(power_iterations(AᴴA))
  end

  # normalization parameters
  regFac = normalize(FISTA, normalizeReg, reg, A, nothing)

  return FISTA(A, AᴴA, vec(reg)[1], x, x₀, xᵒˡᵈ, res, rT(ρ),rT(t),rT(t),iterations,rT(relTol),normalizeReg,regFac,one(rT),rT(Inf),verbose,restart)
end

"""
    init!(it::FISTA, b::vecT
              ; A=solver.A
              , x::vecT=similar(b,0)
              , t::Float64=1.0) where T

(re-) initializes the FISTA iterator
"""
function init!(solver::FISTA{rT,vecT,matA,matAHA}, b::vecT
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

  solver.t = t
  solver.tᵒˡᵈ = t
  # normalization of regularization parameters
  solver.regFac = normalize(solver, solver.normalizeReg, solver.reg, solver.A, solver.x₀)
end

"""
    solve(solver::FISTA, b::Vector)

solves an inverse problem using FISTA.

# Arguments
* `solver::FISTA`                     - the solver containing both system matrix and regularizer
* `b::vecT`                           - data vector
* `A=solver.A`                        - operator for the data-term of the problem
* (`startVector::vecT=similar(b,0)`)  - initial guess for the solution
* (`solverInfo=nothing`)              - solverInfo object

when a `SolverInfo` objects is passed, the residuals are stored in `solverInfo.convMeas`.
"""
function solve(solver::FISTA, b; A=solver.A, startVector=similar(b,0), solverInfo=nothing, kargs...)
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
  solver.x .*= ((1 - solver.tᵒˡᵈ)/solver.t) # here we calculate -α * xᵒˡᵈ, where xᵒˡᵈ is now stored in x
  solver.x .+= ((solver.tᵒˡᵈ-1)/solver.t + 1) .* (solver.xᵒˡᵈ) # add (α+1)*x, where x is now stored in xᵒˡᵈ

  # calculate residuum and do gradient step
  # solver.x .-= solver.ρ .* (solver.AᴴA * solver.x .- solver.x₀)
  mul!(solver.res, solver.AᴴA, solver.x)
  solver.res .-= solver.x₀
  solver.x .-= solver.ρ .* solver.res

  solver.rel_res_norm = norm(solver.res) / solver.norm_x₀
  solver.verbose && println("Iteration $iteration; rel. residual = $(solver.rel_res_norm)")

  # the two lines below are equivalent to the ones above and non-allocating, but require the 5-argument mul! function to implemented for AᴴA, i.e. if AᴴA is LinearOperator, it requires LinearOperators.jl v2
  # mul!(solver.x, solver.AᴴA, solver.xᵒˡᵈ, -solver.ρ, 1)
  # solver.x .+= solver.ρ .* solver.x₀

  # proximal map
  prox!(solver.reg, solver.x; factor = solver.regFac*solver.ρ)

  # gradient restart conditions
  if solver.restart == :gradient
    if real(solver.res ⋅ (solver.x .- solver.xᵒˡᵈ) ) > 0 #if momentum is at an obtuse angle to the negative gradient
      solver.verbose && println("Gradient restart at iter $iteration")
      solver.t = 1
    end
  end

  # predictor-corrector update
  solver.tᵒˡᵈ = solver.t
  solver.t = (1 + sqrt(1 + 4 * solver.tᵒˡᵈ^2)) / 2

  # return the residual-norm as item and iteration number as state
  return solver, iteration+1
end

@inline converged(solver::FISTA) = (solver.rel_res_norm < solver.relTol)

@inline done(solver::FISTA,iteration) = converged(solver) || iteration>=solver.iterations
