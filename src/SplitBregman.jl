export SplitBregman

mutable struct SplitBregman{matT,vecT,opT,ropT,rvecT,preconT} <: AbstractLinearSolver
  # oerators and regularization
  A::matT
  reg::Vector{<:AbstractRegularization}
  regTrafo::Vector{ropT}
  y::vecT
  # fields and operators for x update
  op::opT
  β::vecT
  β_yj::vecT
  y_j::vecT
  # fields for primal & dual variables
  u::vecT
  v::Vector{vecT}
  vᵒˡᵈ::Vector{vecT}
  b::Vector{vecT}
  # other parameters
  precon::preconT
  ρ::rvecT
  iterations::Int64
  iterationsInner::Int64
  iterationsCG::Int64
  # state variables for CG
  cgStateVars::CGStateVariables
  # convergence parameters
  rk::rvecT
  sk::vecT
  eps_pri::rvecT
  eps_dt::vecT
  # eps_dual::Float64
  σᵃᵇˢ::Float64
  absTol::Float64
  relTol::Float64
  tolInner::Float64
  #counter for internal iterations
  iter_cnt::Int64
  normalizeReg::AbstractRegularizationNormalization
end

"""
    SplitBregman(A::matT, x::vecT=zeros(eltype(A),size(A,2))
                ; reg=nothing, regName=["L1","TV"], λ=[0.0,0.0], kargs...) where {matT,vecT}

creates a `SplitBregman` object for the system matrix `A`.

# Arguments
* `A::matT`                     - system matrix
* `x::vecT`                     - Array with the same type and size as the solution
* (`reg=nothing`)               - Regularization object
* (`λ=[0.0]`)                   - Regularization paramters
* (`regTrafo=nothing`)          - transformations applied inside each regularizer
* (`precon=Identity()`)         - preconditionner for the internal CG algorithm
* (`ρ=[1.e2]`)                  - weights for condition on regularized variables
* (`iterations::Int64=10`)      - number of outer iterations
* (`iterationsInner::Int64=50`) - maximum number of inner iterations
* (`iterationsCG::Int64=10`)    - maximum number of CG iterations
* (`absTol::Float64=eps()`)     - abs tolerance for stopping criterion
* (`relTol::Float64=eps()`)     - rel tolerance for stopping criterion
* (`tolInner::Float64=1.e-5`)   - tolerance for CG stopping criterion
"""
function SplitBregman(A::matT, x::vecT=zeros(eltype(A),size(A,2)), b=nothing; λ=[0.0]
                    , reg=L1Regularization(λ[1])
                    , regTrafo=nothing 
                    , precon=Identity()
                    , ρ=[1.e2]
                    , iterations::Int64=10
                    , iterationsInner::Int64=50
                    , iterationsCG::Int64=10
                    , absTol::Float64=eps()
                    , relTol::Float64=eps()
                    , tolInner::Float64=1.e-6
                    , normalizeReg::AbstractRegularizationNormalization = NoNormalization()
                    , kargs...) where {matT, vecT<:AbstractVector}

  if regTrafo == nothing
    regTrafo = [opEye(eltype(x),size(A,2)) for i=1:length(vec(reg))]
  end

  if b==nothing
    y = similar(x,size(A,1))
  else
    y = b
  end

  # operator and fields for the update of x
  op = A'*A
  for i=1:length(vec(reg))
    op += ρ[i]*adjoint(regTrafo[i])*regTrafo[i]
  end
  β = similar(x)
  β_yj = similar(x)
  y_j = similar(x, size(A,1))

  # fields for primal & dual variables
  u = similar(x)
  v = [similar(x, size(regTrafo[i],1)) for i=1:length(vec(reg))]
  vᵒˡᵈ = [similar(v[i]) for i=1:length(vec(reg))]
  b = [similar(v[i]) for i=1:length(vec(reg))]

  # statevariables for CG
  # we store them here to prevent CG from allocating new fields at each call
  statevars = CGStateVariables(zero(u),similar(u),similar(u))

  # convergence parameters
  rk = similar( real.(x), length(vec(reg)) ) #[0.0 for i=1:length(vec(reg))]
  sk = similar(x)
  eps_pri = similar( real.(x), length(vec(reg)) ) # [0.0 for i=1:length(vec(reg))]
  eps_dt = similar(x)

  iter_cnt = 1

  # make sure that ρ is a vector
  if typeof(ρ) <: Real
    ρ_vec = similar(x, real(eltype(x)), 1)
    ρ_vec .= ρ
  else
    ρ_vec = typeof(real.(x))(ρ)
  end

  # normalization parameters
  reg = normalize(SplitBregman, normalizeReg, vec(reg), A, nothing)

  return SplitBregman(A,reg,regTrafo,y,op,β,β_yj,y_j,u,v,vᵒˡᵈ,b,precon,ρ_vec
              ,iterations,iterationsInner,iterationsCG,statevars, rk,sk
              ,eps_pri,eps_dt,0.0,absTol,relTol,tolInner,iter_cnt,normalizeReg)
end

"""
  init!(solver::SplitBregman{matT,vecT,opT,rvecT,preconT}, b::vecT
              ; A::matT=solver.A
              , u::vecT=similar(b,0)
              , kargs...) where {matT,vecT,opT,rvecT,preconT}

(re-) initializes the SplitBregman iterator
"""
function init!(solver::SplitBregman{matT,vecT,opT,ropT,rvecT,preconT}, b::vecT
              ; A::matT=solver.A
              , u::vecT=similar(b,0)
              , kargs...) where {matT,vecT,opT,ropT,rvecT,preconT}

  # operators
  if A != solver.A
    solver.A = A
    solver.op = A'*A
    for i=1:length(vec(reg))
      solver.op += ρ[i]*adjoint(regTrafo[i])*regTrafo[i]
    end
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

  # primal and dual variables
  for i=1:length(solver.reg)
    solver.v[i][:] .= solver.regTrafo[i]*solver.u
    solver.vᵒˡᵈ[i][:] .= 0
    solver.b[i][:] .= 0
  end

  # right hand side for the x-update
  solver.y_j[:] .= b
  solver.β_yj .= adjoint(A) * b

  # convergence parameter
  solver.σᵃᵇˢ = sqrt(length(b))*solver.absTol

  # normalization of regularization parameters
  solver.reg = normalize(solver, solver.normalizeReg, solver.reg, solver.A, b)

  # reset interation counter
  solver.iter_cnt = 1
end

"""
    solve(solver::SplitBregman, b::vecT; kargs...) where vecT

solves an inverse problem using the Split Bregman method.

# Arguments
* `solver::SplitBregman`              - the solver containing both system matrix and regularizer
* `b::vecT`                           - data vector
* (`A::matT=solver.A`)                - operator for the data-term of the problem
* (`startVector::vecT=similar(b,0)`)  - initial guess for the solution
* (`solverInfo=nothing`)              - solverInfo for logging

when a `SolverInfo` objects is passed, the primal residuals `solver.rk`
and the dual residual `norm(solver.sk)` are stored in `solverInfo.convMeas`.
"""
function solve(solver::SplitBregman, b::vecT; A::matT=solver.A, startVector::vecT=similar(b,0), solverInfo=nothing, kargs...) where {vecT,matT}
  # initialize solver parameters
  init!(solver, b; A=A, u=startVector)

  # log solver information
  solverInfo != nothing && storeInfo(solverInfo,solver.v,solver.rk...,norm(solver.sk))

  # perform SplitBregman iterations
  for (iteration, item) = enumerate(solver)
    solverInfo != nothing && storeInfo(solverInfo,solver.v,solver.rk...,norm(solver.sk))
  end

  return solver.u
end

"""
    splitBregman(A, y::vecT, reg::Vector{AbstractRegularization}; kargs...) where vecT

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
  * `y::vecT`                     - data vector (right hand size)
  * `reg::Vector{AbstractRegularization}` - Regularization objects
  * `regTrafo::Vector{Trafo}`     - transformations applied inside each regularizer
  * (`ρ1::Float64=1.0`)           - weighting factor for constraint u=v1
  * (`ρ2::Float64=1.0`)           - weighting factor for constraint u=v2
  * (`precon=Identity()`)         - precondionner to use with CG
  * (`startVector=nothing`)       - start vector
  * (`iterations::Int64=50`)      - maximum number of iterations
  * (`iterationsInner::Int64=50`) - maximum number of inner iterations
  * (`iterationsCG::Int64=10`)    - maximum number of CG iterations
  * (`absTol::Float64=eps()`)     - absolute tolerance for stopping criterion
  * (`relTol::Float64=eps()`)     - relative tolerance for stopping criterion
  * (`tolInner::Float64=1.e-3`)   - relative tolerance for CG
  * (`solverInfo = nothing`)      - `solverInfo` object used to store convergence metrics
"""
function iterate(solver::SplitBregman{matT,vecT,opT,rvecT,preconT}, iteration::Int=1) where {matT,vecT,opT,rvecT,preconT}
  if done(solver, iteration) return nothing end

  # update u
  solver.β[:] .= solver.β_yj
  for i=1:length(solver.reg)
    solver.β[:] .+= solver.ρ[i]*adjoint(solver.regTrafo[i])*(solver.v[i].-solver.b[i])
  end
  cg!(solver.u,solver.op,solver.β,Pl=solver.precon,maxiter=solver.iterationsCG,reltol=solver.tolInner)

  #  proximal map for regularization terms
  for i=1:length(solver.reg)
    copyto!(solver.vᵒˡᵈ[i], solver.v[i])
    solver.v[i][:] .= solver.regTrafo[i]*solver.u .+ solver.b[i]
    if solver.ρ[i] != 0
      prox!(solver.reg[i], solver.v[i], λ(solver.reg[i])/solver.ρ[i])
    end
  end

  # update b
  for i=1:length(solver.reg)
    solver.b[i] .+= solver.regTrafo[i]*solver.u .- solver.v[i]
  end

  # update convergence criteria
  # primal residuals norms (one for each constraint)
  for i=1:length(solver.reg)
    solver.rk[i] = norm(solver.regTrafo[i]*solver.u-solver.v[i])
    solver.eps_pri[i] = solver.σᵃᵇˢ + solver.relTol*max( norm(solver.regTrafo[i]*solver.u), norm(solver.v[i]) )
  end
  # accumulated dual residual
  # effectively this corresponds to combining all constraints into one larger constraint.
  solver.sk[:] .= 0.0
  solver.eps_dt[:] .= 0.0
  for i=1:length(solver.reg)
    solver.sk[:] .+= solver.ρ[i]*adjoint(solver.regTrafo[i])*(solver.v[i].-solver.vᵒˡᵈ[i])
    solver.eps_dt[:] .+= solver.ρ[i]*adjoint(solver.regTrafo[i])*solver.b[i]
  end

  if update_y(solver,iteration)
    solver.y_j[:] .+= solver.y .- solver.A*solver.u
    solver.β_yj[:] .= adjoint(solver.A) * solver.y_j
    # reset v and b
    for i=1:length(solver.reg)
      solver.v[i][:] .= solver.regTrafo[i]*solver.u
      solver.b[i] .= 0
    end
    solver.iter_cnt += 1
    iteration = 0
  end

  return solver.rk[1], iteration+1

end

function converged(solver::SplitBregman)
  if norm(solver.sk) >= solver.σᵃᵇˢ+solver.relTol*norm(solver.eps_dt)
    return false
  else
    for i=1:length(solver.reg)
      (solver.rk[i] >= solver.eps_pri[i]) && return false
    end
  end

  return true
end

@inline done(solver::SplitBregman,iteration::Int) = (iteration==1 && solver.iter_cnt>solver.iterations)

function update_y(solver::SplitBregman,iteration::Int)
  conv = converged(solver)
  return conv || iteration >= solver.iterationsInner
end
