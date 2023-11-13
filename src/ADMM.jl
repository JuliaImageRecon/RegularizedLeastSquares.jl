export admm, ADMM

mutable struct ADMM{rT,matT,opT,R,ropT,P,vecT,rvecT,preconT} <: AbstractPrimalDualSolver where {vecT <: AbstractVector{Union{rT, Complex{rT}}}, rvecT <: AbstractVector{rT}}
  # operators and regularization
  A::matT
  reg::Vector{R}
  regTrafo::Vector{ropT}
  proj::Vector{P}
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
  normalizeReg::AbstractRegularizationNormalization
  vary_ρ::Symbol
  verbose::Bool
end

"""
    ADMM(A, x; kwargs...)

creates an `ADMM` object for the system matrix `A`.

# Arguments
* `A`                           - system matrix
* `x`                     - (optional) array with the same type and size as the solution

# Keywords
* `reg`          - regularization term vector
* `normalizeReg`         - regularization normalization scheme
* `precon=Identity()`        - preconditionner for the internal CG algorithm
* `ρ::Real=1.e-2`          - penalty of the augmented lagrangian
* `vary_ρ::Bool=:none`      - vary rho to balance primal and dual feasibility
* `iterations::Int64=50`      - max number of ADMM iterations
* `iterationsInner::Int64=10` - max number of internal CG iterations
* `absTol::Real=eps()`     - abs tolerance for stopping criterion
* `relTol::Real=eps()`     - rel tolerance for stopping criterion
* `tolInner::Real=1.e-5`   - rel tolerance for CG stopping criterion

See also [`createLinearSolver`](@ref), [`solve`](@ref).
"""
function ADMM(A::matT, x::Vector{T}=zeros(eltype(A),size(A,2));
              reg=[L1Regularization(zero(eltype(A)))]
            , AᴴA::opT=nothing
            , precon=Identity()
            , ρ=1e-1
            , iterations::Integer=50
            , iterationsInner::Integer=10
            , absTol::Real=eps(real(T))
            , relTol::Real=eps(real(T))
            , tolInner::Real=1e-5
            , normalizeReg::AbstractRegularizationNormalization = NoNormalization()
            , vary_ρ::Symbol=:none
            , verbose::Bool=false
            , kargs...) where {T,matT,opT, }
  # TODO: The constructor is not type stable

  # unify Floating types
  absTol = real(T)(absTol)
  relTol = real(T)(relTol)
  tolInner = real(T)(tolInner)

  reg = vec(reg) # using a custom method of vec(.)

  regTrafo = []
  indices = findsinks(AbstractProjectionRegularization, reg)
  proj = [reg[i] for i in indices]
  proj = identity.(proj)
  deleteat!(reg, indices)
  # Retrieve constraint trafos
  for r in reg
    trafoReg = findfirst(ConstraintTransformedRegularization, r)
    if isnothing(trafoReg) 
      push!(regTrafo, opEye(eltype(x),size(A,2)))
    else
      push!(regTrafo, trafoReg)
    end
  end
  regTrafo = identity.(regTrafo)
  
  if typeof(ρ) <: Number
    ρ_vec = [real(T).(ρ) for i = 1:length(reg)]
  else
    ρ_vec = real(T).(ρ)
  end

  xᵒˡᵈ = similar(x)

  # fields for primal & dual variables
  z = [similar(x, size(A,2)) for i=1:length(reg)]
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

  # normalization parameters
  reg = normalize(ADMM, normalizeReg, reg, A, nothing)

  return ADMM(A,reg,regTrafo,proj,AᴴA,β,β_y,x,xᵒˡᵈ,z,zᵒˡᵈ,u,uᵒˡᵈ,precon,ρ_vec,iterations
              ,iterationsInner,statevars, rᵏ,sᵏ,ɛᵖʳⁱ,ɛᵈᵘᵃ,zero(real(T)),Δ,absTol,relTol,tolInner
              ,normalizeReg, vary_ρ, verbose)
end

"""
  init!(solver::ADMM{matT,opT,vecT,rvecT,preconT}, b::vecT
              ; A::matT=solver.A
              , AᴴA::opT=solver.AᴴA
              , x::vecT=similar(b,0)
              , kargs...) where {matT,opT,vecT,rvecT,preconT}

(re-) initializes the ADMM iterator
"""
function init!(solver::ADMM{rT,matT,opT,R,ropT,P,vecT,rvecT,preconT}, b::vecT
              ; A::matT=solver.A
              , AᴴA::opT=solver.AᴴA
              , x::vecT=similar(b,0)
              , kargs...) where {rT,matT,opT,R,ropT,P,vecT,rvecT,preconT}

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
  solver.reg = normalize(solver, solver.normalizeReg, solver.reg, solver.A, b)

end

"""
    solve(solver::ADMM, b; kwargs...) where {matT,vecT}

solves an inverse problem using ADMM.

# Arguments
* `solver::ADMM`                  - the solver containing both system matrix and regularizer
* `b::Vector`                     - data vector

# Keywords
* `A::matT=solver.A`            - operator for the data-term of the problem
* `startVector::Vector{T}=T[]`  - initial guess for the solution
* `solverInfo=nothing`          - solverInfo for logging

when a `SolverInfo` objects is passed, the primal residuals `solver.rᵏ`
and the dual residual `norm(solver.sᵏ)` are stored in `solverInfo.convMeas`.
"""
function solve(solver::ADMM, b; A=solver.A, AᴴA=solver.AᴴA, startVector=similar(b,0), solverInfo=nothing, kargs...)
  # initialize solver parameters
  init!(solver, b; A=A, AᴴA=AᴴA, x=startVector)

  # log solver information
  solverInfo != nothing && storeInfo(solverInfo,solver.x,solver.rᵏ...,solver.sᵏ...)

  # perform ADMM iterations
  for (iteration, item) = enumerate(solver)
    solverInfo != nothing && storeInfo(solverInfo,solver.x,solver.rᵏ...,solver.sᵏ...)
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

  for proj in solver.proj
    prox!(proj, solver.x)
  end

  for i=1:length(solver.reg)
    # 2. update z using the proximal map of 1/ρ*g(x)
    solver.zᵒˡᵈ[i] .= solver.z[i]
    solver.z[i] .= solver.regTrafo[i]*solver.x .+ solver.u[i]
    if solver.ρ[i] != 0
      prox!(solver.reg[i], solver.z[i], λ(solver.reg[i])/solver.ρ[i])
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
