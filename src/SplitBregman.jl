export SplitBregman

mutable struct SplitBregman{matT,opT,T,preconT} <: AbstractLinearSolver
  # oerators and regularization
  A::matT
  reg::Vector{Regularization}
  y::Vector{T}
  # fields and operators for x update
  op::opT
  β::Vector{T}
  β_yj::Vector{T}
  y_j::Vector{T}
  # fields for primal & dual variables
  u::Vector{T}
  v::Vector{T}
  w::Vector{T}
  vᵒˡᵈ::Vector{T}
  wᵒˡᵈ::Vector{T}
  bv::Vector{T}
  bw::Vector{T}
  # other parameters
  precon::preconT
  μ::Float64
  ν::Float64
  ρ::Float64
  iterations::Int64
  iterationsInner::Int64
  iterationsCG::Int64
  # state variables for CG
  cgStateVars::CGStateVariables
  # convergence parameters
  rk_1::Float64
  rk_2::Float64
  sk::Float64
  eps_pri_1::Float64
  eps_pri_2::Float64
  eps_dual::Float64
  σᵃᵇˢ::Float64
  absTol::Float64
  relTol::Float64
  tolInner::Float64
  #counter for internal iterations
  iter_cnt::Int64
end

"""
    SplitBregman(A; reg=nothing, regName=["L1","TV"], λ=[0.0,0.0], kargs...)

creates a `SplitBregman` object for the system matrix `A`.

# Arguments
* `A` - system matrix
* (`reg=nothing`)               - Regularization object
* (`regName=["L1","TV"]`)       - name of the regularizations to use (if reg==nothing)
* (`λ=[0.0, 0.0]`)              - Regularization paramters
* (`precon=Identity()`)         - preconditionner for the internal CG algorithm
* (`μ::Float64=1.e2`)           - weight for the data term
* (`ν::Float64=1.e2`)           - weight for condition on 1. regularized variable
* (`ρ::Float64=1.e2`)           - weight for condition on 2. regularized variable
* (`iterations::Int64=10`)      - number of outer iterations
* (`iterationsInner::Int64=50`) - maximum number of inner iterations
* (`iterationsCG::Int64=10`)    - maximum number of CG iterations
* (`absTol::Float64=1.e-8`)     - abs tolerance for stopping criterion
* (`relTol::Float64=1.e-6`)     - rel tolerance for stopping criterion
* (`tolInner::Float64=1.e-5`)   - tolerance for CG stopping criterion
"""
function SplitBregman(A, b=nothing; reg=nothing, regName=["L1","TV"]
                    , λ=[0.0,0.0]
                    , precon=Identity()
                    , μ::Float64=1.e2
                    , ν::Float64=1.e2
                    , ρ::Float64=1.e2
                    , iterations::Int64=10
                    , iterationsInner::Int64=50
                    , iterationsCG::Int64=10
                    , absTol::Float64=1.e-8
                    , relTol::Float64=1.e-6
                    , tolInner::Float64=1.e-6
                    , kargs...) where {matT,opT}

  if reg == nothing
    reg = Regularization(regName, λ, kargs...)
  end
  if b==nothing
    y = zeros(eltype(A),size(A,1))
  else
    y = b
  end

  # operator and fields for the update of x
  op = μ*A'*A + (ν+ρ)*opEye(size(A,2))
  β = zeros(eltype(A),size(A,2))
  β_yj = zeros(eltype(A),size(A,2))
  y_j = zeros(eltype(A),size(A,1))

  # fields for primal & dual variables
  u = zeros(eltype(A),size(A,2))
  v = zeros(eltype(A),size(A,2))
  vᵒˡᵈ = zeros(eltype(A),size(A,2))
  w = zeros(eltype(A),size(A,2))
  wᵒˡᵈ = zeros(eltype(A),size(A,2))
  bv = zeros(eltype(A),size(A,2))
  bw = zeros(eltype(A),size(A,2))

  # statevariables for CG
  # we store them here to prevent CG from allocating new fields at each call
  statevars = CGStateVariables(zero(u),similar(u),similar(u))

  iter_cnt = 1

  return SplitBregman(A,vec(reg),y,op,β,β_yj,y_j,u,v,w,vᵒˡᵈ,wᵒˡᵈ,bv,bw,precon,μ,ν,ρ
              ,iterations,iterationsInner,iterationsCG,statevars, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,absTol,relTol,tolInner,iter_cnt)
end

"""
  init!(solver::SplitBregman{matT,opT,T,preconT}
              ; A::matT=solver.A
              , b::Vector{T}=T[]
              , u::Vector{T}=T[]
              , kargs...) where {matT,opT,T,preconT}

(re-) initializes the SplitBregman iterator
"""
function init!(solver::SplitBregman{matT,opT,T,preconT}
              ; A::matT=solver.A
              , b::Vector{T}=solver.y
              , u::Vector{T}=T[]
              , kargs...) where {matT,opT,T,preconT}

  # operators
  if A != solver.A
    solver.A = A
    solver.op = μ*A'*A + (solver.ν+solver.ρ)*opEye(length(u))
  end
  solver.y = b

  # start vector
  if isempty(u)
    if !isempty(b)
      solver.u[:] .= adjoint(A) * b
    else
      solver.u[:] .= 0.0
    end
  else
    solver.u[:] .= u
  end
  solver.v[:] .= 0
  solver.w[:] .= 0

  # right hand side for the x-update
  solver.y_j[:] .= b
  solver.β_yj .= solver.μ * adjoint(A) * b

  # primal and dual variables

  solver.vᵒˡᵈ[:] .= 0
  solver.wᵒˡᵈ[:] .= 0
  solver.bv[:] .= 0
  solver.bw[:] .= 0

  # convergence parameter
  solver.σᵃᵇˢ = sqrt(length(b))*solver.absTol

  # reset interation counter
  solver.iter_cnt = 1
end

"""
    solve(solver::SplitBregman, b::Vector)

solves an inverse problem using the Split Bregman method.

# Arguments
* `solver::SplitBregman`          - the solver containing both system matrix and regularizer
* `b::Vector`                     - data vector
* (`A::matT=solver.A`)            - operator for the data-term of the problem
* (`startVector::Vector{T}=T[]`)  - initial guess for the solution
* (`solverInfo=nothing`)          - solverInfo for logging
"""
function solve(solver::SplitBregman, b::Vector{T}; A::matT=solver.A, startVector::Vector{T}=T[], solverInfo=nothing, kargs...) where {matT,T}
  # initialize solver parameters
  init!(solver; A=A, b=b, u=startVector)

  # log solver information
  solverInfo != nothing && storeInfo(solverInfo,solver.A,b,solver.v;xᵒˡᵈ=solver.vᵒˡᵈ,reg=solver.reg)

  # perform SplitBregman iterations
  for (iteration, item) = enumerate(solver)
    solverInfo != nothing && storeInfo(solverInfo,solver.A,b,solver.v;xᵒˡᵈ=solver.vᵒˡᵈ,reg=solver.reg)
  end

  return solver.u
end

"""
    splitBregman(A, y::Vector, reg::Vector{Regularization}; kargs...)

Split Bregman method

Solve the problem: min_x r1(x) + λ*r2(x) such that Ax=b
Here:
  x: variable (vector)
  b: measured data
  A: a general linear operator
  g(X): a convex but not necessarily a smooth function

For details see:
T. Goldstein, S. Osher ,
The Split Bregman Method for l1 Regularized Problems

  # Arguments
  * `A`                           - system matrix
  * `y::Vector`                   - data vector (right hand size)
  * `reg::Vector{Regularization}` - Regularization objects
  * (`λ::Float64=1.e-2`)          - factor to enforce constrained with resp. to w (reg[1])
  * (`μ::Float64=1.e-2`)          - factor to enforce constrained with resp. to v (reg[2])
  * (`ρ::Float64=1.e-2`)          - factor to enforce constrained with resp. to y
  * (`precon=Identity()`)         - precondionner to use with CG
  * (`startVector=nothing`)       - start vector
  * (`iterations::Int64=50`)      - maximum number of iterations
  * (`iterationsInner::Int64=50`) - maximum number of inner iterations
  * (`iterationsCG::Int64=10`) - maximum number of CG iterations
  * (`absTol::Float64=1.e-8`)     - absolute tolerance for stopping criterion
  * (`relTol::Float64=1.e-6`)     - relative tolerance for stopping criterion
  * (`tolInner::Float64=1.e-3`)   - tolerance for CG
  * (`solverInfo = nothing`)      - `solverInfo` object used to store convergence metrics
"""
function iterate(solver::SplitBregman{matT,opT,T,preconT}, iteration::Int=1) where {matT,opT,T,preconT}
  if done(solver, iteration) return nothing end

  # update u
  solver.β[:] .= solver.β_yj .+ solver.ν*(solver.w.-solver.bw) .+ solver.ρ*(solver.v.-solver.bv)
  cg!(solver.u,solver.op,solver.β,Pl=solver.precon,maxiter=solver.iterationsCG,tol=solver.tolInner)

  #  proximal map for 1. regularization
  copyto!(solver.wᵒˡᵈ, solver.w) #solver.wᵒˡᵈ[:] .= solver.w
  solver.w[:] .= solver.u .+ solver.bw
  if solver.ν != 0
    solver.reg[1].prox!(solver.w,1.0/solver.ν; solver.reg[1].params...)
    # reg[1].prox!(w,reg[1].λ/solver.ν; reg[1].params...)
  end

  # proximal map for 2. regularization
  copyto!(solver.vᵒˡᵈ, solver.v)
  solver.v[:]  .= solver.u .+ solver.bv
  if solver.ρ != 0
    solver.reg[2].prox!(solver.v,solver.reg[2].λ/solver.ρ; solver.reg[2].params...)
  end

  # update bv and bw
  # solver.bv = solver.bv + solver.u - solver.v
  # solver.bw = solver.bw + solver.u - solver.w
  solver.bv .+= solver.u .- solver.v
  solver.bw .+= solver.u .- solver.w


  solver.rk_1 = norm(solver.u-solver.v)
  solver.rk_2 = norm(solver.u-solver.w)
  solver.eps_pri_1 = solver.σᵃᵇˢ + solver.relTol*max( norm(solver.u), norm(solver.v) )
  solver.eps_pri_2 = solver.σᵃᵇˢ + solver.relTol*max( norm(solver.u), norm(solver.w) )
  solver.sk = norm(solver.ρ*(solver.v.-solver.vᵒˡᵈ) + solver.ν*(solver.w.-solver.wᵒˡᵈ))
  solver.eps_dual = solver.σᵃᵇˢ + solver.relTol*norm(solver.ρ*solver.v.+solver.ν*solver.w)

  if update_y(solver,iteration)
    solver.y_j[:] .+= solver.y .- solver.A*solver.u
    solver.β_yj[:] .= solver.μ * adjoint(solver.A) * solver.y_j
    solver.iter_cnt += 1
    iteration = 0
  end

  return solver.rk_1, iteration+1

end

@inline converged(solver::SplitBregman) = (solver.rk_1<solver.eps_pri_1 && solver.rk_2<solver.eps_pri_2 && solver.sk<solver.eps_dual)

@inline done(solver::SplitBregman,iteration::Int) = (iteration==1 && solver.iter_cnt>solver.iterations)

function update_y(solver::SplitBregman,iteration::Int)
  conv = converged(solver)
  return conv || iteration >= solver.iterationsInner
end
