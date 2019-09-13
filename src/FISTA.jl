export fista

mutable struct FISTA{matT,T} <: AbstractLinearSolver
  A::matT
  reg::Regularization
  x::Vector{T}
  xᵒˡᵈ::Vector{T}
  res::Vector{T}
  res_norm::Float64
  res_norm_old::Float64
  ρ::Float64
  t::Float64
  tᵒˡᵈ::Float64
  iterations::Int64
  relTol::Float64
end

"""
    FISTA(A; reg=nothing, regName=["L1"], λ=[0.0], kargs...)

creates a `FISTA` object for the system matrix `A`.

# Arguments
* `A` - system matrix
* (`reg=nothing`)           - Regularization object
* (`regName=["L1"]`)        - name of the Regularization to use (if reg==nothing)
* (`λ=[0.0]`)               - Regularization paramter
* (`ρ::Float64=1`)          - step size for gradient step
* (`t::Float64=1.0`)        - parameter for predictor-corrector step
* (`relTol::Float64=1.e-5`) - tolerance for stopping criterion
* (`iterations::Int64=50`)  - maximum number of iterations
"""
function FISTA(A::matT; reg=nothing, regName=["L1"]
              , λ=[0.0]
              , ρ::Float64=1.0
              , t::Float64=1.0
              , relTol::Float64=1.e-5
              , iterations::Int64=50
              , kargs...) where matT

  if reg == nothing
    reg = Regularization(regName, λ, kargs...)
  end

  x = zeros(eltype(A),size(A,2))
  xᵒˡᵈ = zeros(eltype(A),size(A,2))
  res = zeros(eltype(A),size(A,1))

  return FISTA(A,vec(reg)[1],x,xᵒˡᵈ,res,0.0,0.0,ρ,t,t,iterations,relTol)
end

"""
    init!(it::FISTA{matT,T}
              ; A::matT=solver.A
              , b::Vector{T}=zeros(T,size(A,1))
              , x::Vector{T}=zeros(T,size(A,1))
              , t::Float64=1.0) where T

(re-) initializes the FISTA iterator
"""
function init!(solver::FISTA{matT,T}
              ; A::matT=solver.A
              , b::Vector{T}=T[]
              , x::Vector{T}=T[]
              , t::Float64=1.0
              ) where {matT,T}

  solver.A = A
  if isempty(x)
    if !isempty(b) #!iszero(b)
      solver.x[:] .= adjoint(A) * b
    else
      solver.x[:] .= zeros(T,size(A,2))
    end
  end
  solver.xᵒˡᵈ[:] .= solver.x  # this could also be zero
  solver.res[:] .= A*solver.x-b
  solver.res_norm_old = 0.0
  solver.res_norm = norm(solver.res)
  solver.t = t
  solver.tᵒˡᵈ = t
end

"""
    solve(solver::FISTA, b::Vector)

solves an inverse problem using FISTA.

# Arguments
* `solver::FISTA`                 - the solver containing both system matrix and regularizer
* `b::Vector`                     - data vector
* `A::matT=solver.A`            - operator for the data-term of the problem
* (`startVector::Vector{T}=T[]`)  - initial guess for the solution
* (`solverInfo=nothing`)          - solverInfo object
"""
function solve(solver::FISTA, b::Vector{T}; A::matT=solver.A, startVector::Vector{T}=T[], solverInfo=nothing, kargs...) where {matT,T}
  # initialize solver parameters
  init!(solver; A=A, b=b, x=startVector)

  # log solver information
  solverInfo != nothing && storeInfo(solverInfo,solver.A,b,solver.x;xᵒˡᵈ=solver.xᵒˡᵈ,reg=[solver.reg],residual=solver.res)

  # perform FISTA iterations
  for (iteration, item) = enumerate(solver)
    solverInfo != nothing && storeInfo(solverInfo,solver.A,b,solver.x;xᵒˡᵈ=solver.xᵒˡᵈ,reg=[solver.reg],residual=solver.res)
  end

  return solver.x
end

"""
  iterate(it::FISTA, iteration::Int=0)

performs one fista iteration.
"""
function iterate(solver::FISTA{matT,T}, iteration::Int=0) where {matT,T}
  if done(solver, iteration) return nothing end

  # gradient step
  solver.xᵒˡᵈ[:] .= solver.x
  solver.x[:] .= solver.x - solver.ρ* (solver.A' * solver.res)

  # proximal map
  solver.reg.prox!(solver.x, solver.ρ*solver.reg.λ; solver.reg.params...)

  # predictor-corrector update
  solver.tᵒˡᵈ = solver.t
  solver.t = (1. + sqrt(1. + 4. * solver.tᵒˡᵈ^2)) / 2.
  solver.x[:] .= solver.x + (solver.tᵒˡᵈ-1)/solver.t*(solver.x-solver.xᵒˡᵈ)

  # update residual
  # solver.res .= solver.A*solver.x-solver.b
  solver.res .+= solver.A*(solver.x-solver.xᵒˡᵈ)  # this relies on a proper initialization in init()
  solver.res_norm_old = solver.res_norm
  solver.res_norm = norm(solver.res)

  # return the residual-norm as item and iteration number as state
  return solver.res_norm, iteration+1
end

@inline converged(solver::FISTA) = ( abs(solver.res_norm-solver.res_norm_old)/solver.res_norm_old < solver.relTol )

@inline done(solver::FISTA,iteration::Int) = converged(solver) || iteration>=solver.iterations
