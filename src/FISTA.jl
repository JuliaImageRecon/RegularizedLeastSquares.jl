export FISTA

mutable struct FISTA{rT <: Real, vecT <: Union{AbstractVector{rT}, AbstractVector{Complex{rT}}}, matA, matAHA} <: AbstractLinearSolver
  A::matA
  AHA::matAHA
  reg::Regularization
  x::vecT
  x₀::vecT
  xᵒˡᵈ::vecT
  res::vecT
  ρ::rT
  t::rT
  tᵒˡᵈ::rT
  iterations::Int64
  relTol::rT
  normalizeReg::Bool
  regFac::rT
  norm_x₀::rT
  rel_res_norm::rT
  verbose::Bool
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
* (`AHA=A'*A`)              - specialized normal operator, default is `A'*A`
* (`λ=0`)                   - regularization parameter
* (`ρ=0.95`)                - step size for gradient step
* (`normalize_ρ=false`)     - normalize step size by the maximum eigenvalue of `AHA`
* (`t=1.0`)                 - parameter for predictor-corrector step
* (`relTol::Float64=1.e-5`) - tolerance for stopping criterion
* (`iterations::Int64=50`)  - maximum number of iterations
"""
function FISTA(
              ; A=nothing
              , AHA=A'*A
              , x::AbstractVector{T}=Vector{eltype(AHA)}(undef,size(AHA,2))
              , reg=nothing
              , regName=["L1"]
              , λ=[zero(real(eltype(x)))]
              , ρ=0.95
              , normalize_ρ=true
              , t=1
              , relTol=eps(real(eltype(x)))
              , iterations=50
              , normalizeReg=false
              , verbose = false
              , kargs...) where {T}

  rT = real(T)
  if reg === nothing
    reg = Regularization(regName, λ, kargs...)
  end

  x₀   = similar(x)
  xᵒˡᵈ = similar(x)
  res  = similar(x)
  res[1] = Inf # avoid spurious convergence in first iterations

  if normalize_ρ
    ρ /= abs(power_iterations(AHA))
  end

  return FISTA(A, AHA, vec(reg)[1], x, x₀, xᵒˡᵈ, res, rT(ρ),rT(t),rT(t),iterations,rT(relTol),normalizeReg,one(rT),one(rT),rT(Inf),verbose)
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

  if solver.A === nothing
    solver.x₀ = b
  else
    solver.x₀ .= adjoint(solver.A) * b
  end
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
  if solver.normalizeReg
    solver.regFac = norm(solver.x₀,1)/length(solver.x₀)
  else
    solver.regFac = 1
  end
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
  solverInfo != nothing && storeInfo(solverInfo,solver.x,norm(solver.res))

  # perform FISTA iterations
  for (iteration, item) = enumerate(solver)
    solverInfo != nothing && storeInfo(solverInfo,solver.x,norm(solver.res))
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
  solver.reg.prox!(solver.x, solver.regFac*solver.ρ*solver.reg.λ; solver.reg.params...)

  # predictor-corrector update
  solver.tᵒˡᵈ = solver.t
  solver.t = (1 + sqrt(1 + 4 * solver.tᵒˡᵈ^2)) / 2

  # return the residual-norm as item and iteration number as state
  return solver, iteration+1
end

@inline converged(solver::FISTA) = (solver.rel_res_norm < solver.relTol)

@inline done(solver::FISTA,iteration) = converged(solver) || iteration>=solver.iterations
