export admm, ADMM

mutable struct ADMM{rT,matT,opT,ropT,vecT,rvecT,preconT, R} <: AbstractLinearSolver where {vecT <: AbstractVector{Union{rT, Complex{rT}}}, rvecT <: AbstractVector{rT}, R <: AbstractRegularization}
  # operators and regularization
  A::matT
  reg::Vector{R}
  regTrafo::Vector{ropT}
  # fields and operators for x update
  AᴴA::opT
  β::vecT
  β_y::vecT
  # fields for primal & dual variables
  x::vecT
  xᵒˡᵈ::vecT
  z::Vector{vecT}
  zᵒˡᵈ::Vector{vecT}
  u::Vector{vecT}
  uᵒˡᵈ::Vector{vecT}
  # other parameters
  precon::preconT
  ρ::rvecT # TODO: Switch all these vectors to Tuple
  iterations::Int64
  iterationsInner::Int64
  # state variables for CG
  cgStateVars::CGStateVariables
  # convergence parameters
  rᵏ::rvecT
  sᵏ::rvecT
  ɛᵖʳⁱ::rvecT
  ɛᵈᵘᵃ::rvecT
  σᵃᵇˢ::rT
  Δ::rvecT
  absTol::rT
  relTol::rT
  tolInner::rT
  normalizeReg::Bool
  regFac::rT
  vary_ρ::Symbol
  verbose::Bool
end

"""
    ADMM(A, x::vecT=zeros(eltype(A),size(A,2))
          ; reg=nothing, regName=["L1"], λ=[0.0], kargs...)

creates an `ADMM` object for the system matrix `A`.

# Arguments
* `A`                           - system matrix
* `x::vecT`                     - Array with the same type and size as the solution
* (`reg=nothing`)               - Regularization object
* (`λ=[0.0]`)                   - Regularization paramter
* (`regTrafo=nothing`)          - transformations applied inside each regularizer
* (`precon=Identity()`)         - preconditionner for the internal CG algorithm
* (`ρ::Real=1.e-2`)          - penalty of the augmented lagrangian
* (`adaptRho::Bool=false`)      - adapt rho to balance primal and dual feasibility
* (`iterations::Int64=50`)      - max number of ADMM iterations
* (`iterationsInner::Int64=10`) - max number of internal CG iterations
* (`absTol::Real=eps()`)     - abs tolerance for stopping criterion
* (`relTol::Real=eps()`)     - rel tolerance for stopping criterion
* (`tolInner::Real=1.e-5`)   - rel tolerance for CG stopping criterion
"""
function ADMM(A::matT, x::Vector{T}=zeros(eltype(A),size(A,2)); reg=L1Regularization(λ[1])
            , λ=[0.0]
            , regTrafo=nothing
            , AᴴA::opT=nothing
            , precon=Identity()
            , ρ=1e-1
            , iterations::Integer=50
            , iterationsInner::Integer=10
            , absTol::Real=eps(real(T))
            , relTol::Real=eps(real(T))
            , tolInner::Real=1e-5
            , normalizeReg::Bool=false
            , vary_ρ::Symbol=:none
            , verbose::Bool=false
            , kargs...) where {T,matT,opT}
  # TODO: The constructor is not type stable

  # unify Floating types
  if typeof(ρ) <: Number
    ρ_vec = [real(T).(ρ)]
  else
    ρ_vec = real(T).(ρ)
  end
  λ = real(T).(λ)
  absTol = real(T)(absTol)
  relTol = real(T)(relTol)
  tolInner = real(T)(tolInner)

  reg = vec(reg) # using a custom method of vec(.)

  if regTrafo == nothing
    regTrafo = [opEye(eltype(x),size(A,2)) for _=1:length(reg)]
  end

  xᵒˡᵈ = similar(x)

  # fields for primal & dual variables
  z = [similar(x, size(regTrafo[i],1)) for i=1:length(reg)]
  zᵒˡᵈ = [similar(z[i]) for i=1:length(reg)]
  u = [similar(z[i]) for i=1:length(reg)]
  uᵒˡᵈ = [similar(u[i]) for i=1:length(reg)]

  # operator and fields for the update of x
  if AᴴA == nothing
    AᴴA = A'*A
  end
  β = similar(x)
  β_y = similar(x)

  # statevariables for CG
  # we store them here to prevent CG from allocating new fields at each call
  statevars = CGStateVariables(zero(x),similar(x),similar(x))

  # convergence parameters
  rᵏ   = Array{real(T)}(undef, length(reg))
  sᵏ   = similar(rᵏ)
  ɛᵖʳⁱ = similar(rᵏ)
  ɛᵈᵘᵃ = similar(rᵏ)
  Δ    = similar(rᵏ)

  return ADMM(A,reg,regTrafo,AᴴA,β,β_y,x,xᵒˡᵈ,z,zᵒˡᵈ,u,uᵒˡᵈ,precon,ρ_vec,iterations
              ,iterationsInner,statevars, rᵏ,sᵏ,ɛᵖʳⁱ,ɛᵈᵘᵃ,zero(real(T)),Δ,absTol,relTol,tolInner
              ,normalizeReg,one(real(T)), vary_ρ, verbose)
end

"""
  init!(solver::ADMM{matT,opT,vecT,rvecT,preconT}, b::vecT
              ; A::matT=solver.A
              , AᴴA::opT=solver.AᴴA
              , x::vecT=similar(b,0)
              , kargs...) where {matT,opT,vecT,rvecT,preconT}

(re-) initializes the ADMM iterator
"""
function init!(solver::ADMM{rT,matT,opT,ropT,vecT,rvecT,preconT}, b::vecT
              ; A::matT=solver.A
              , AᴴA::opT=solver.AᴴA
              , x::vecT=similar(b,0)
              , kargs...) where {rT,matT,opT,ropT,vecT,rvecT,preconT}

  # operators
  if A != solver.A
    solver.A = A
    solver.AᴴA = ( AᴴA!=nothing ? AᴴA : A'*A )
  end

  # start vector
  if isempty(x)
    solver.x .= 0
 else
    solver.x[:] .= x
 end

  # primal and dual variables
  for i=1:length(solver.reg)
    solver.z[i] .= solver.regTrafo[i]*solver.x
    solver.u[i] .= 0
  end

  # right hand side for the x-update
  solver.β_y[:] .= adjoint(A) * b

  # convergence parameter
  solver.rᵏ .= Inf
  solver.sᵏ .= Inf
  solver.ɛᵖʳⁱ .= 0
  solver.ɛᵈᵘᵃ .= 0
  solver.σᵃᵇˢ = sqrt(length(b))*solver.absTol
  solver.Δ .= Inf

  # normalization of regularization parameters
  if solver.normalizeReg
    solver.regFac = norm(b,1)/length(b)
  else
    solver.regFac = 1
  end
end

"""
    solve(solver::ADMM, b::vecT
          ; A::matT=solver.A
          , startVector::vecT=similar(b,0)
          , solverInfo=nothing
          , kargs...) where {matT,vecT}

solves an inverse problem using ADMM.

# Arguments
* `solver::ADMM`                  - the solver containing both system matrix and regularizer
* `b::Vector`                     - data vector
* (`A::matT=solver.A`)            - operator for the data-term of the problem
* (`startVector::Vector{T}=T[]`)  - initial guess for the solution
* (`solverInfo=nothing`)          - solverInfo for logging

when a `SolverInfo` objects is passed, the primal residuals `solver.rᵏ`
and the dual residual `norm(solver.sᵏ)` are stored in `solverInfo.convMeas`.
"""
function solve(solver::ADMM{rT,matT,opT,ropT,vecT,rvecT,preconT}, b::vecT; A=solver.A, AᴴA=solver.AᴴA, startVector::vecT=similar(b,0), solverInfo=nothing, kargs...) where {rT,matT,opT,ropT,vecT,rvecT,preconT}
  # initialize solver parameters
  init!(solver, b; A=A, AᴴA=AᴴA, x=startVector)

  # log solver information
  solverInfo != nothing && storeInfo(solverInfo,solver.z,solver.rᵏ...,solver.sᵏ...)

  # perform ADMM iterations
  for (iteration, item) = enumerate(solver)
    solverInfo != nothing && storeInfo(solverInfo,solver.z,solver.rᵏ...,solver.sᵏ...)
  end

  return solver.x
end

"""
  iterate(it::ADMM, iteration::Int=0)

performs one ADMM iteration.
"""
function iterate(solver::ADMM, iteration::Integer=0)
  if done(solver, iteration) return nothing end
  solver.verbose && println("Outer ADMM Iteration #$iteration")

  # 1. solve arg min_x 1/2|| Ax-b ||² + ρ/2 Σ_i||Φi*x+ui-zi||²
  # <=> (A'A+ρ Σ_i Φi'Φi)*x = A'b+ρΣ_i Φi'(zi-ui)
  solver.β .= solver.β_y
  AᴴA = solver.AᴴA
  for i=1:length(solver.reg)
    solver.β[:] .+= solver.ρ[i]*adjoint(solver.regTrafo[i])*(solver.z[i].-solver.u[i])
    AᴴA += solver.ρ[i] * adjoint(solver.regTrafo[i]) * solver.regTrafo[i]
  end
  solver.verbose && println("conjugated gradients: ")
  solver.xᵒˡᵈ .= solver.x
  cg!(solver.x, AᴴA, solver.β, Pl=solver.precon
      , maxiter=solver.iterationsInner, reltol=solver.tolInner, statevars=solver.cgStateVars, verbose = solver.verbose)

  for i=1:length(solver.reg)
    # 2. update z using the proximal map of 1/ρ*g(x)
    solver.zᵒˡᵈ[i] .= solver.z[i]
    solver.z[i] .= solver.regTrafo[i]*solver.x .+ solver.u[i]
    if solver.ρ[i] != 0
      prox!(solver.reg[i], solver.z[i], solver.regFac*solver.reg[i].λ/solver.ρ[i])
    end

    # 3. update u
    solver.uᵒˡᵈ[i] .= solver.u[i]
    solver.u[i] .+= solver.regTrafo[i]*solver.x .- solver.z[i]

    # update convergence measures (one for each constraint)
    solver.rᵏ[i] = norm(solver.regTrafo[i]*solver.x-solver.z[i])  # primal residual (x-z)
    solver.sᵏ[i] = norm(solver.ρ[i] * adjoint(solver.regTrafo[i]) * (solver.z[i] .- solver.zᵒˡᵈ[i])) # dual residual (concerning f(x))

    solver.ɛᵖʳⁱ[i] = max(norm(solver.regTrafo[i]*solver.x), norm(solver.z[i]))
    solver.ɛᵈᵘᵃ[i] = norm(solver.ρ[i] * adjoint(solver.regTrafo[i]) * solver.u[i])

    Δᵒˡᵈ = solver.Δ[i]
    solver.Δ[i] = norm(solver.x    .- solver.xᵒˡᵈ   ) +
                  norm(solver.z[i] .- solver.zᵒˡᵈ[i]) +
                  norm(solver.u[i] .- solver.uᵒˡᵈ[i])

    if (solver.vary_ρ == :balance && solver.rᵏ[i]/solver.ɛᵖʳⁱ[i] > 10solver.sᵏ[i]/solver.ɛᵈᵘᵃ[i]) || # adapt ρ according to Boyd et al.
       (solver.vary_ρ == :PnP     && solver.Δ[i]/Δᵒˡᵈ > 0.9) # adapt ρ according to Chang et al.
      solver.ρ[i] *= 2
      solver.u[i] ./= 2
    elseif solver.vary_ρ == :balance && solver.sᵏ[i]/solver.ɛᵈᵘᵃ[i] > 10solver.rᵏ[i]/solver.ɛᵖʳⁱ[i]
      solver.ρ[i] /= 2
      solver.u[i] .*= 2
    end

    if solver.verbose
      println("rᵏ[$i] = $(solver.rᵏ[i])")
      println("sᵏ[$i] = $(solver.sᵏ[i])")
      println("ɛᵖʳⁱ[$i] = $(solver.ɛᵖʳⁱ[i])")
      println("ɛᵈᵘᵃ[$i] = $(solver.ɛᵈᵘᵃ[i])")
      println("Δᵒˡᵈ = $(Δᵒˡᵈ)")
      println("Δ[$i] = $(solver.Δ[i])")
      println("Δ/Δᵒˡᵈ = $(solver.Δ[i]/Δᵒˡᵈ)")
      println("current ρ[$i] = $(solver.ρ[i])")
      flush(stdout)
    end
  end

  # return the primal feasibility measure as item and iteration number as state
  return solver.rᵏ, iteration+1
end

function converged(solver::ADMM)
  for i=1:length(solver.reg)
    (solver.rᵏ[i] >= solver.σᵃᵇˢ + solver.relTol * solver.ɛᵖʳⁱ[i]) && return false
    (solver.sᵏ[i] >= solver.σᵃᵇˢ + solver.relTol * solver.ɛᵈᵘᵃ[i]) && return false
  end
  return true
end

@inline done(solver::ADMM,iteration::Int) = converged(solver) || iteration>=solver.iterations
