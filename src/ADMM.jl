export admm

mutable struct ADMM{matT,opT,T,preconT} <: AbstractLinearSolver
  # oerators and regularization
  A::matT
  reg::Regularization
  # fields and operators for x update
  op::opT
  β::Vector{T}
  # fields for primal & dual variables
  x::Vector{T}
  z::Vector{T}
  zᵒˡᵈ::Vector{T}
  u::Vector{T}
  # other parameters
  precon::preconT
  ρ::Float64
  ρ0::Float64
  adaptRho::Bool
  iterations::Int64
  iterationsInner::Int64
  # state variables for CG
  cgStateVars::CGStateVariables
  # convergence parameters
  rᵏ::Float64
  sᵏ::Float64
  ɛᵖʳⁱ::Float64
  ɛᴰᵘᵃˡ::Float64
  σᵃᵇˢ::Float64
  absTol::Float64
  relTol::Float64
  tolInner::Float64
end

"""
    ADMM(A; reg=nothing, regName=["L1"], λ=[0.0], kargs...)

creates an `ADMM` object for the system matrix `A`.

# Arguments
* `A` - system matrix
* (`reg=nothing`)               - Regularization object
* (`regName=["L1"]`)            - name of the Regularization to use (if reg==nothing)
* (`λ=[0.0]`)                   - Regularization paramter
* (`precon=Identity()`)         - preconditionner for the internal CG algorithm
* (`ρ::Float64=1.e-2`)          - penalty of the augmented lagrangian
* (`adaptRho::Bool=false`)      - adapt rho to balance primal and dual feasibility
* (`iterations::Int64=50`)      - max number of ADMM iterations
* (`iterationsInner::Int64=10`) - max number of internal CG iterations
* (`absTol::Float64=1.e-8`)     - abs tolerance for stopping criterion
* (`relTol::Float64=1.e-6`)     - rel tolerance for stopping criterion
* (`tolInner::Float64=1.e-5`)   - tolerance for CG stopping criterion
"""
function ADMM(A::matT; reg=nothing, regName=["L1"]
            , λ=[0.0]
            , AHA::opT=nothing
            , precon=Identity()
            , ρ::Float64=1.e-2
            , adaptRho::Bool=false
            , iterations::Int64=50
            , iterationsInner::Int64=10
            , absTol::Float64=1.e-8
            , relTol::Float64=1.e-6
            , tolInner::Float64=1.e-5
            , kargs...) where {matT,opT}

  if reg == nothing
    reg = Regularization(regName, λ, kargs...)
  end

  # operator and fields for the update of x
  if AHA != nothing
    op = AHA
  else
    op = A'*A
  end
  β = zeros(eltype(A),size(A,2))

  # fields for primal & dual variables
  x = zeros(eltype(A),size(A,2))
  z = zeros(eltype(A),size(A,2))
  zᵒˡᵈ = zeros(eltype(A),size(A,2))
  u = zeros(eltype(A),size(A,2))

  # statevariables for CG
  # we store them here to prevent CG from allocating new fields at each call
  statevars = CGStateVariables(zero(x),similar(x),similar(x))
  return ADMM(A,vec(reg)[1],op,β,x,z,zᵒˡᵈ,u,precon,ρ,ρ,adaptRho
              ,iterations,iterationsInner,statevars, 0.0,0.0,0.0,0.0,0.0,absTol,relTol,tolInner)
end

"""
  init!(solver::ADMM{matT,opT,T,preconT}
              ; A::matT=solver.A
              , AHA::opT=solver.op
              , b::Vector{T}=T[]
              , x::Vector{T}=T[]
              , kargs...) where {matT,opT,T,preconT}

(re-) initializes the ADMM iterator
"""
function init!(solver::ADMM{matT,opT,T,preconT}
              ; A::matT=solver.A
              , AHA::opT=solver.op
              , b::Vector{T}=T[]
              , x::Vector{T}=T[]
              , kargs...) where {matT,opT,T,preconT}

  # operators
  if A != solver.A
    solver.A = A
    if AHA != nothing
      solver.op = AHA
    else
      solver.op = A'*A
    end
  end

  # start vector
  if isempty(x)
    if !isempty(b)
      solver.x[:] .= adjoint(A) * b
      solver.z[:] .= 0.0
    else
      solver.x[:] .= 0.0
      solver.z[:] .= 0.0
    end
  else
    solver.x[:] .= x
    solver.z[:] .= x
  end

  # right hand side for the x-update
  solver.β .= adjoint(A)*b

  # primal and dual variables
  solver.zᵒˡᵈ[:] .= 0
  solver.u .= 0

  # set ρ
  solver.ρ=solver.ρ0

  # convergence parameter
  solver.σᵃᵇˢ = sqrt(length(b))*solver.absTol
end

"""
    solve(solver::ADMM, b::Vector{T}
          ; A::matT=solver.A, startVector::Vector{T}=T[]
          , startVector::Vector{T}=T[], solverInfo=nothing
          , kargs...) where {matT,T}

solves an inverse problem using ADMM.

# Arguments
* `solver::ADMM`                  - the solver containing both system matrix and regularizer
* `b::Vector`                     - data vector
* (`A::matT=solver.A`)            - operator for the data-term of the problem
* (`startVector::Vector{T}=T[]`)  - initial guess for the solution
* (`solverInfo=nothing`)          - solverInfo for logging
"""
function solve(solver::ADMM, b::Vector{T}; A::matT=solver.A, startVector::Vector{T}=T[], solverInfo=nothing, kargs...) where {matT,T}
  # initialize solver parameters
  init!(solver; A=A, b=b, x=startVector)

  # log solver information
  solverInfo != nothing && storeInfo(solverInfo,solver.A,b,solver.z;xᵒˡᵈ=solver.zᵒˡᵈ,reg=[solver.reg])

  # perform ADMM iterations
  for (iteration, item) = enumerate(solver)
    solverInfo != nothing && storeInfo(solverInfo,solver.A,b,solver.z;xᵒˡᵈ=solver.zᵒˡᵈ,reg=[solver.reg])
  end

  return solver.x
end

"""
  iterate(it::ADMM, iteration::Int=0)

performs one ADMM iteration.
"""
function iterate(solver::ADMM{matT,opT,T,preconT}, iteration::Int=0) where {matT,opT,T,preconT}
  if done(solver, iteration) return nothing end

  # 1. solve arg min_x 1/2|| Ax-b ||² + ρ/2 ||x+u-z||²
  # <=> (A'A+ρ)*x = A'b+ρ(z-u)
  cg!(solver.x, solver.op+solver.ρ*opEye(length(solver.x))
      , solver.β+solver.ρ*(solver.z-solver.u), Pl=solver.precon
      , maxiter=solver.iterationsInner, tol=solver.tolInner, statevars=solver.cgStateVars)
  # 2. update z using the proximal map of 1/ρ*g(x)
  copyto!(solver.zᵒˡᵈ, solver.z)
  solver.z[:] .= solver.x .+ solver.u
  solver.reg.prox!(solver.z, solver.reg.λ/solver.ρ; solver.reg.params...)

  # 3. update u
  solver.u[:] .+= solver.x .- solver.z

  # update convergence measures
  solver.rᵏ = norm(solver.x-solver.z)  # primal residual (x-z)
  solver.ɛᵖʳⁱ = solver.σᵃᵇˢ + solver.relTol*max( norm(solver.x), norm(solver.z) )
  solver.sᵏ = norm(solver.ρ * (solver.z .- solver.zᵒˡᵈ)) # dual residual (concerning f(x))
  solver.ɛᴰᵘᵃˡ = solver.σᵃᵇˢ + solver.relTol*norm(solver.ρ*solver.u);

  # adapt ρ to given residuals
  if solver.adaptRho
    if solver.rᵏ > 10.0*solver.sᵏ
      solver.ρ = 2.0*solver.ρ
    elseif solver.sᵏ > 10.0*solver.rᵏ
      solver.ρ = solver.ρ/2.0
    end
  end

  # return the primal feasibilty measure as item and iteration number as state
  return solver.rᵏ, iteration+1
end

@inline converged(solver::ADMM) = (solver.rᵏ<solver.ɛᵖʳⁱ && solver.sᵏ < solver.ɛᴰᵘᵃˡ)

@inline done(solver::ADMM,iteration::Int) = converged(solver) || iteration>=solver.iterations
