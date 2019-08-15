export admm

mutable struct ADMM <: AbstractLinearSolver
  A
  regularizer::Regularization
  params
end

"""
    ADMM(A; reg=nothing, regName=["L1"], λ=[0.0], kargs...)

creates an `ADMM` object for the system matrix `A`.

# Arguments
* `A` - system matrix
* (`reg=nothing`)     - Regularization object
* (`regName=["L1"]`)  - name of the Regularization to use (if reg==nothing)
* (`λ=[0.0]`)         - Regularization paramter
"""
function ADMM(A; reg=nothing, regName=["L1"], λ=[0.0], kargs...)

  if reg == nothing
    reg = Regularization(regName, λ, kargs...)
  end

  return ADMM(A,vec(reg)[1],kargs)
end

"""
    solve(solver::ADMM, b::Vector)

solves an inverse problem using ADMM.

# Arguments
* `solver::ADMM`  - the solver containing both system matrix and regularizer
* `b::Vector`     - data vector
"""
function solve(solver::ADMM, b::Vector)
  if get(solver.params, :accelerate, false)
    return fadmm(solver.A, b, solver.regularizer; solver.params...)
  else
    return admm(solver.A, b, solver.regularizer; solver.params...)
  end
end

"""
    admm(A, b::Vector, reg::Regularization; kargs...)

Alternating Direction Method of Multipliers

Solve the problem: X = arg min_x 1/2*|| Ax-b||² + λ*g(X) where:
  x: variable (vector)
  b: measured data
  A: a general linear operator
  g(X): a convex but not necessarily a smooth function

For details see:
Boyd et al.,
Distributed Optimization and Statistical Learning via the Alternating Direction
  Method of Multipliers,
Foundations and Trends in Machine Learning, Vol. 3, No. 1 (2010) 1–122

# Arguments
* `A`                           - system matrix
* `b::Vector`                   - data vector (right hand size)
* `reg::Regularization`         - Regularization object
* (`AHA=nothing`)               - normal operator adjoint(A)*A
* (`ρ::Float64=1.e-2`)          - scaling factor to enforce constrained in augm. Lagrangian
* (`precon=Identity()`)         - precondionner to use with CG
* (`startVector=nothing`)       - start vector
* (`iterations::Int64=50`)      - maximum number of iterations
* (`iterationsInner::Int64=10`) - maximum number of CG iterations
* (`absTol::Float64=1.e-8`)     - absolute tolerance for stopping criterion
* (`relTol::Float64=1.e-6`)     - relative tolerance for stopping criterion
* (`tolInner::Float64=1.e-3`)   - tolerance for CG
* (`adaptRho::Bool=false`)      - if true, `rho` is adapted to balance primal and dual residuals
* (`solverInfo = nothing`)      - `solverInfo` object used to store convergence metrics
"""
function admm(A, b::Vector, reg::Regularization
              ; AHA=nothing
              , ρ::Float64=1.e-2
              , precon=Identity()
              , startVector=nothing
              , iterations::Int64=50
              , iterationsInner::Int64=10
              , absTol::Float64=1.e-8
              , relTol::Float64=1.e-6
              , tolInner::Float64=1.e-5
              , adaptRho::Bool=false
              , solverInfo = nothing
              , kargs...)

  σᵃᵇˢ = sqrt(length(b))*absTol # rescaled tolerance
  # initialize x, u and z
  x = zeros(eltype(b),size(A,2))
  z = zeros(eltype(x), size(x))
  if startVector == nothing
    x[:] = A' * b
  else
    x[:] = copy(startVector)
    z[:] = x
  end
  xᵒˡᵈ = zeros(eltype(x), size(x))
  zᵒˡᵈ = zeros(eltype(x), size(x))
  u = zeros(eltype(x), size(x))

  if AHA!=nothing
    op = AHA+ρ*opEye(length(x))
  else
    op = A'*A+ρ*opEye(length(x))
  end

  solverInfo != nothing && storeInfo(solverInfo,A,b,x;xᵒˡᵈ=xᵒˡᵈ,reg=[reg])

  β = A' * b

  @showprogress 1 "Computing..." for k=1:iterations
    # 1. solve arg min_x 1/2|| Ax-b ||² + ρ/2 ||x+u-z||²
    # <=> (A'A+ρ)*x = A'b+ρ(z-u)
    xᵒˡᵈ[:] = x[:]
    cg!(x,A'*A+ρ*opEye(length(x)),β+ρ*(z-u),Pl=precon,maxiter=iterationsInner,tol=tolInner)

    # 2. update z using the proximal map of 1/ρ*g(x)
    zᵒˡᵈ[:] = z
    z[:]=x[:]+u[:]
    reg.prox!(z, reg.λ/ρ; reg.params...)

    # 3. update u
    u[:]=u+x-z

    solverInfo != nothing && storeInfo(solverInfo,A,b,x;xᵒˡᵈ=xᵒˡᵈ,reg=[reg])

    # exit if residual is below tolerance
    rᵏ = norm(x-z)  # primal residual (x-z)
    ɛᵖʳⁱ = σᵃᵇˢ + relTol*max( norm(x), norm(z) );
    sᵏ = norm(ρ * (z - zᵒˡᵈ)) # dual residual (concerning f(x))
    ɛᴰᵘᵃˡ = σᵃᵇˢ + relTol*norm(ρ*u);

    if (rᵏ < ɛᵖʳⁱ) && (sᵏ < ɛᴰᵘᵃˡ)
      @info "ADMM converged at iteration $(k)"
      break;
    end

    # adapt ρ to given residuals
    if adaptRho
      τ=2.0
      μ=10.0
      if rᵏ > μ*sᵏ
        ρ = τ*ρ
      elseif sᵏ > μ*rᵏ
        ρ = ρ/τ
      end
    end
  end

  return x
end

# fast version which emplois a Nesterov-type acceleration
function fadmm(A, b::Vector, reg::Regularization
              ; ρ::Float64=1.e-2
              , AHA=nothing
              , precon=Identity()
              , startVector=nothing
              , iterations::Int64=50
              , iterationsInner::Int64=10
              , absTol::Float64=1.e-8
              , relTol::Float64=1.e-6
              , tolInner::Float64=1.e-5
              , adaptRho::Bool=false
              , solverInfo = nothing
              , kargs...)

  σᵃᵇˢ = sqrt(length(b))*absTol # rescaled tolerance
  η=0.999 # parameter for restart criterion

  # initialize x, u and z
  x = zeros(eltype(b),size(A,2))
  if startVector == nothing
    x[:] = A' * b
  else
    x[:] = copy(startVector)
  end
  xᵒˡᵈ = copy(x)
  z = copy(x)
  ẑ = copy(z)
  zᵒˡᵈ = copy(z)
  u = zeros(eltype(x), size(x))
  û = copy(u)
  uᵒˡᵈ = copy(u)

  if AHA!=nothing
    op = AHA+ρ*opEye(length(x))
  else
    op = A'*A+ρ*opEye(length(x))
  end

  solverInfo != nothing && storeInfo(solverInfo,A,b,x;xᵒˡᵈ=xᵒˡᵈ,reg=[reg])

  β = A' * b

  α = 1.0
  c = Inf
  @showprogress 1 "Computing..." for k=1:iterations
    # 1. solve arg min_x 1/2|| Ax-b ||² + ρ/2 ||x+û-ẑ||²
    # <=> (A'A+ρ)*x = A'b+ρ(z-u)
    xᵒˡᵈ[:] = x[:]
    cg!(x,A'*A+ρ*opEye(length(x)),β+ρ*(ẑ-û),Pl=precon,maxiter=iterationsInner,tol=tolInner)

    # 2. update z using the proximal map of 1/ρ*g(x)
    zᵒˡᵈ[:] = z
    z[:]=x[:]+û[:]
    reg.prox!(z, reg.λ/ρ; reg.params...)

    # 3. update u
    uᵒˡᵈ[:] = u
    u[:]=û+x-z

    # check if combined residual decreases
    cᵒˡᵈ = c
    c = ρ*norm(u-û)^2 + ρ*norm(z-ẑ)
    if c < η*cᵒˡᵈ
      # apply Nesterov type acceleration
      αᵒˡᵈ = α
      α = 0.5*(1.0 + sqrt(1.0 + 4.0 * αᵒˡᵈ^2))
      ẑ[:] = z + (αᵒˡᵈ-1)/α*(z-zᵒˡᵈ)
      û[:] = u + (αᵒˡᵈ-1)/α*(u-uᵒˡᵈ)
    else
      # restart
      α = 1
      ẑ[:] = zᵒˡᵈ
      û[:] = uᵒˡᵈ
      c = cᵒˡᵈ/η
    end

    solverInfo != nothing && storeInfo(solverInfo,A,b,x;xᵒˡᵈ=xᵒˡᵈ,reg=[reg])

    # exit if residual is below tolerance
    rᵏ = norm(x-z)  # primal residual (x-z)
    ɛᵖʳⁱ = σᵃᵇˢ + relTol*max( norm(x), norm(z) );
    sᵏ = norm(ρ * (z - zᵒˡᵈ)) # dual residual (concerning f(x))
    ɛᴰᵘᵃˡ = σᵃᵇˢ + relTol*norm(ρ*u);
    if (rᵏ < ɛᵖʳⁱ) && (sᵏ < ɛᴰᵘᵃˡ)
      @info "FADMM converged at iteration $(k)"
      break;
    end

    # adapt ρ to given residuals
    if adaptRho
      τ=2.0
      μ=10.0
      if rᵏ > μ*sᵏ
        ρ = τ*ρ
      elseif sᵏ > μ*rᵏ
        ρ = ρ/τ
      end
    end
  end

  return x
end
