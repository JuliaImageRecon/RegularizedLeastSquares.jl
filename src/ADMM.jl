export admm

mutable struct ADMM <: AbstractLinearSolver
  A
  regularizer::Regularization
  params
end

ADMM(A, regularization; kargs...) = ADMM(A,regularization,kargs)

# function solve(solver::ADMM, b::Vector)
#   return admm(solver.A, b, solver.regularizer; solver.params...)
# end

function solve(solver::ADMM, b::Vector)
  if get(solver.params, :accelerate, false)
    return fadmm(solver.A, b, solver.regularizer; solver.params...)
  else
    return admm(solver.A, b, solver.regularizer; solver.params...)
  end
end

"""
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
"""
function admm(A, b::Vector, reg::Regularization
              ; AHA=nothing
              , precon=nothing
              , startVector=nothing
              , iterations::Int64=50
              , iterationsInner::Int64=10
              , ρ::Float64=1.e-2
              , ɛᵃᵇˢ::Float64=1.e-8
              , ɛʳᵉˡ::Float64=1.e-6
              , solverInfo = nothing
              , kargs...)

  σᵃᵇˢ = sqrt(length(b))*ɛᵃᵇˢ
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
    if precon != nothing
      x[:] = cg(op, x, β+ρ*(z-u), precon, iterations=iterationsInner )
    else
      x[:] = cg(op, x, β+ρ*(z-u), iterations=iterationsInner)
    end

    # 2. update z using the proximal map of 1/ρ*g(x)
    zᵒˡᵈ[:] = z
    z[:]=x[:]+u[:]
    reg.prox!(z, reg.λ/ρ; reg.params...)

    # 3. update u
    u[:]=u+x-z

    solverInfo != nothing && storeInfo(solverInfo,A,b,x;xᵒˡᵈ=xᵒˡᵈ,reg=[reg])

    # exit if residual is below tolerance
    rᵏ = norm(x-z)
    ɛᵖʳⁱ = σᵃᵇˢ + ɛʳᵉˡ*max( norm(x), norm(z) );
    sᵏ = norm(ρ * (z - zᵒˡᵈ))
    ɛᴰᵘᵃˡ = σᵃᵇˢ + ɛʳᵉˡ*norm(ρ*u);

    if (rᵏ < ɛᵖʳⁱ) && (sᵏ < ɛᴰᵘᵃˡ)
      @info "ADMM converged at iteration $(k)"
      break;
    end
  end

  return x
end

# fast version which emplois a Nesterov-type acceleration
function fadmm(A, b::Vector, reg::Regularization
              ; AHA=nothing
              , startVector=nothing
              , iterations::Int64=50
              , iterationsInner::Int64=10
              , ρ::Float64=1.e-2
              , η::Float64=0.999
              , ɛᵃᵇˢ::Float64=1.e-8
              , ɛʳᵉˡ::Float64=1.e-6
              , solverInfo = nothing
              , kargs...)

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
    x[:] = cg(op, x,  β+ρ*(ẑ-û), iterations=iterationsInner)

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
    rᵏ = norm(x-z)
    ɛᵖʳⁱ = σᵃᵇˢ + ɛʳᵉˡ*max( norm(x), norm(z) );
    sᵏ = norm(ρ * (z - zᵒˡᵈ))
    ɛᴰᵘᵃˡ = σᵃᵇˢ + ɛʳᵉˡ*norm(ρ*u);
    if (rᵏ < ɛᵖʳⁱ) && (sᵏ < ɛᴰᵘᵃˡ)
      @info "FADMM converged at iteration $(k)"
      break;
    end
  end

  return x
end
