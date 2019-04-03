export splitBregman


mutable struct SplitBregman <: AbstractLinearSolver
  A
  reg::Vector{Regularization}
  params
end

function SplitBregman(A; reg=nothing, regName=["L1","TV"], λ=[0.0,0.0], kargs...)

  if reg == nothing
    reg = Regularization(regName, λ, kargs...)
  end

  return SplitBregman(A, vec(reg), kargs)
end

function solve(solver::SplitBregman, b::Vector)
  return splitBregman(solver.A, b, solver.reg; solver.params...)
end

"""
 Split Bregman method

 Solve the problem: min_x r1(x) + λ*r2(x) such that Ax=b
 Here:
    x: variable (vector)
    b: measured data
    A: a general linear operator
    g(X): a convex but not necessarily a smooth function

  For details see:
  T. Goldstein, S. Osher ,
  The Split Bregman Method for l1 Regularized Problems,
"""
function splitBregman(A, y::Vector, reg::Vector{Regularization}
              ; λ::Float64=1.e2
              , μ::Float64=1.e2
              , ρ::Float64=1.e2
              , precon=Identity()
              , startVector=nothing
              , iterations::Int64=10
              , iterationsInner::Int64=50
              , iterationsCG::Int64=10
              , absTol::Float64=1.e-8
              , relTol::Float64=1.e-6
              , tolInner::Float64=1.e-3
              , solverInfo = nothing
              , kargs...)

  σᵃᵇˢ = sqrt(size(A,1))*absTol

  # initialize variables
  u = A' * y
  uᵒˡᵈ = zeros(eltype(u),size(u))
  nu = length(u)
  v = zeros(eltype(u), size(u))
  vᵒˡᵈ = zeros(eltype(u),size(u))
  w = zeros(eltype(u), size(u))
  wᵒˡᵈ = zeros(eltype(u),size(u))
  bv = zeros(eltype(u), size(v))
  bw = zeros(eltype(u), size(w))
  β = zeros(eltype(u), size(v))

  # normal operator
  op = μ*A'*A + (λ+ρ)*opEye(length(u))

  # some precalculations
  β_yj = copy(β)
  y_j = copy(y)
  Φw = zeros(eltype(u), size(w))

  for j = 1:iterations
    uᵒˡᵈ[:] .= u
    β_yj[:] .= μ * A' * y_j
    # solve constrained subproblem
    for k = 1:iterationsInner
      # update u analytically
      β[:] .= β_yj .+ λ*(w-bw) .+ ρ*(v-bv)
      if isa(precon, Identity)
        u[:] = cg(op,u,β,iterations=iterationsCG,relTol=tolInner,solverInfo=solverInfo,storeIterations=true)
      else
        u[:] = cg(op,u,β,precon,iterations=iterationsCG,relTol=tolInner,solverInfo=solverInfo,storeIterations=true)
      end

      #  proximal map for L1 regularization
      wᵒˡᵈ[:] .= w
      w[:] .= u .+ bw
      if λ != 0
        reg[1].prox!(w,1.0/λ; reg[1].params...)
      end

      # proximal map for LR regularization
      vᵒˡᵈ .= v
      v[:]  .= u .+ bv
      if ρ != 0
        reg[2].prox!(v,reg2.λ/ρ; reg[2].params...)
      end

      # update bv and bw
      bv = bv + u -v
      bw = bw + u -w

      solverInfo != nothing && storeInfo(solverInfo,A,y,u;xᵒˡᵈ=uᵒˡᵈ,reg=reg)

      rk_1 = norm(u-v)
      rk_2 = norm(u-w)
      eps_pri_1 = σᵃᵇˢ + relTol*max( norm(u), norm(v) );
      eps_pri_2 = σᵃᵇˢ + relTol*max( norm(u), norm(w) );
      sk = norm(ρ*(v - vᵒˡᵈ) + λ*(w-wᵒˡᵈ))
      eps_dual = σᵃᵇˢ + relTol*norm(ρ*v+λ*w);

      if (rk_1 < eps_pri_1) && (rk_2 < eps_pri_2) && (sk < eps_dual)
        break;
      end

    end

    # update data of subproblems
    y_j[:] = y_j + y - A*u
  end

  return u

end
