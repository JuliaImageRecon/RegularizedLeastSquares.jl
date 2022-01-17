export fista

mutable struct FISTA{rT <: Real, vecT <: Union{AbstractVector{rT}, AbstractVector{Complex{rT}}}, matT} <: AbstractLinearSolver where {rT}
  A::matT
  AᴴA::matT
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
end

"""
    FISTA(A::matT, x::vecT=zeros(eltype(A),size(A,2))
          ; reg=nothing, regName=["L1"], λ=[0.0], kargs...) where {matT,vecT}

creates a `FISTA` object for the system matrix `A`.

# Arguments
* `A`                       - system matrix
* `x::vecT`                 - array with the same type and size as the solution
* (`reg=nothing`)           - regularization object
* (`regName=["L1"]`)        - name of the Regularization to use (if reg==nothing)
* (`AᴴA=A'*A`)              - specialized normal operator, default is `A'*A`
* (`λ=0`)                   - regularization paramter
* (`ρ=1`)                   - step size for gradient step
* (`normalize_ρ=false`)     - normalize step size by the maximum eigenvalue of `AᴴA`
* (`t=1.0`)                 - parameter for predictor-corrector step
* (`relTol::Float64=1.e-5`) - tolerance for stopping criterion
* (`iterations::Int64=50`)  - maximum number of iterations
"""
function FISTA(A, x::AbstractVector{T}=Vector{eltype(A)}(undef,size(A,2)); reg=nothing, regName=["L1"]
              , AᴴA=A'*A
              , λ=0
              , ρ=0.95
              , normalize_ρ=true
              , t=1
              , relTol=eps(real(T))
              , iterations=50
              , normalizeReg=false
              , kargs...) where {rT <: Real, T <: Union{rT, Complex{rT}}}

  if reg == nothing
    reg = Regularization(regName, λ, kargs...)
  end

  x₀   = similar(x)
  xᵒˡᵈ = similar(x)
  res  = similar(x)
  res[1] = Inf # avoid spurious convergence in first iterations

  if normalize_ρ
    ρ /= abs(power_iterations(AᴴA))
  end

  return FISTA(A, AᴴA, vec(reg)[1], x, x₀, xᵒˡᵈ, res, rT(ρ),rT(t),rT(t),iterations,rT(relTol),normalizeReg,one(rT),one(rT))
end

"""
    init!(it::FISTA{matT,T}, b::vecT
              ; A::matT=solver.A
              , x::vecT=similar(b,0)
              , t::Float64=1.0) where T

(re-) initializes the FISTA iterator
"""
function init!(solver::FISTA{rT,vecT,matT}, b::vecT
              ; x::vecT=similar(b,0)
              , t=1
              ) where {rT,vecT,matT}

  solver.x₀ .= adjoint(solver.A) * b
  solver.norm_x₀ = norm(solver.x₀)

  if isempty(x)
    solver.x .= solver.ρ .* solver.x₀
  else
    solver.x .= x
  end

  solver.t = t
  solver.tᵒˡᵈ = t
  # normalization of regularization parameters
  if solver.normalizeReg
    solver.regFac = norm(b,1)/length(b)
  else
    solver.regFac = 1
  end
end

"""
    solve(solver::FISTA, b::Vector)

solves an inverse problem using FISTA.

# Arguments
* `solver::FISTA`                 - the solver containing both system matrix and regularizer
* `b::vecT`                     - data vector
* `A::matT=solver.A`            - operator for the data-term of the problem
* (`startVector::vecT=similar(b,0)`)  - initial guess for the solution
* (`solverInfo=nothing`)          - solverInfo object

when a `SolverInfo` objects is passed, the residuals are stored in `solverInfo.convMeas`.
"""
function solve(solver::FISTA, b::vecT; A::matT=solver.A, startVector::vecT=similar(b,0), solverInfo=nothing, kargs...) where {matT,vecT}
  # initialize solver parameters
  init!(solver, b; x=startVector)

  # log solver information
  solverInfo != nothing && storeInfo(solverInfo,solver.x,solver.res_norm)

  # perform FISTA iterations
  for (iteration, item) = enumerate(solver)
    solverInfo != nothing && storeInfo(solverInfo,solver.x,solver.res_norm)
  end

  return solver.x
end

"""
  iterate(it::FISTA{matT,vecT}, iteration::Int=0) where {matT,vecT}

performs one fista iteration.
"""
function iterate(solver::FISTA{matT,vecT}, iteration::Int=0) where {matT,vecT}
  if done(solver, iteration) return nothing end

  solver.xᵒˡᵈ .= solver.x

  # calculate residuum and do gradient step
  # solver.x .-= solver.ρ .* (solver.AᴴA * solver.x .- solver.x₀)
  mul!(solver.res, solver.AᴴA, solver.xᵒˡᵈ)
  solver.res .-= solver.x₀
  solver.x .-= solver.ρ .* solver.res

  # the two lines below are equivalent to the ones above and non-allocating, but require the 5-argument mul! function to implemented for AᴴA, i.e. if AᴴA is LinearOperator, it requires LinearOperators.jl v2
  # mul!(solver.x, solver.AᴴA, solver.xᵒˡᵈ, -solver.ρ, 1)
  # solver.x .+= solver.ρ .* solver.x₀

  # proximal map
  solver.reg.prox!(solver.x, solver.regFac*solver.ρ*solver.reg.λ; solver.reg.params...)

  # predictor-corrector update
  solver.tᵒˡᵈ = solver.t
  solver.t = (1 + sqrt(1 + 4 * solver.tᵒˡᵈ^2)) / 2
  solver.x .+= ((solver.tᵒˡᵈ-1)/solver.t) .* (solver.x .- solver.xᵒˡᵈ)

  # return the residual-norm as item and iteration number as state
  return solver, iteration+1
end

@inline converged(solver::FISTA) = (norm(solver.res) / solver.norm_x₀ < solver.relTol)

@inline done(solver::FISTA,iteration) = converged(solver) || iteration>=solver.iterations
