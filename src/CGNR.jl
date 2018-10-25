export cgnr


mutable struct CGNR <: AbstractLinearSolver
  A
  reg::Regularization
  params
end

CGNR(A, reg; kargs...) = CGNR(A,reg,kargs)

function solve(solver::CGNR, u::Vector)
  return cgnr(solver.A, u; lambd=solver.reg.params[:lambdL2], solver.params... )
end



"""
This funtion implements the cgnr algorithm.
"""
function cgnr(S::AbstractMatrix{T}, u::Vector{T};
iterations = 10, lambd::Real = 0.0, startVector = nothing, weights = nothing,
enforceReal = false, enforcePositive = false, sparseTrafo = nothing,
solverInfo = nothing, kargs... ) where T
  N = size(S,2)
  M = size(S,1)

  if startVector == nothing
    cl = zeros(T,N)     #solution vector
  else
    cl = startVector
  end
  rl = zeros(T,M)     #residual vector
  zl = zeros(T,N)     #temporary vector
  pl = zeros(T,N)     #temporary vector
  vl = zeros(T,M)     #temporary vector
  xl = zeros(T,M)     #temporary vector
  αl = zero(T)        #temporary scalar
  βl = zero(T)        #temporary scalar
  ζl = zero(T)        #temporary scalar

  #pre iteration
  #rₗ = u - Sᵗ*cₗ
  copyto!(rl,u)
  gemv!('N',-one(T), S, cl, one(T), rl)
  #zₗ = Sᶜ*rₗ, where ᶜ denotes complex conjugation
  if weights != nothing
    xl = rl .* weights
    gemv!('C',one(T), S, xl, zero(T), zl)
  else
    gemv!('C',one(T), S, rl, zero(T), zl)
  end
  #pₗ = zₗ
  copyto!(pl,zl)
  #start iteration
  for l=1:min(iterations,size(S,2))
    #vₗ = Sᵗ*pₗ
    gemv!('N',one(T), S, pl, zero(T), vl)

    # αₗ = zₗᴴ⋅zₗ/(vₗᴴ⋅vₗ+λ*pₗᴴ⋅pₗ)
    ζl = norm(zl)^2
    normvl = weights == nothing ? dot(vl,vl) : dot(vl,weights.*vl)
    lambd > 0 ? αl = ζl/(normvl+lambd*norm(pl)^2) : αl = ζl/normvl

    #cₗ += αₗ*pₗ
    BLAS.axpy!(αl,pl,cl)

    if solverInfo != nothing
      push!( solverInfo.xNorm, norm(cl))
      push!( solverInfo.resNorm, norm(rl) )
    end

    #rₗ += -αₗ*vₗ
    BLAS.axpy!(-αl,vl,rl)

    #zₗ = Sᶜ*rₗ-lambd*cₗ
    if weights != nothing
      xl = rl .* weights
      gemv!('C',one(T), S, xl, zero(T), zl)
    else
      gemv!('C',one(T), S, rl, zero(T), zl)
    end
    if lambd > 0
      BLAS.axpy!(-lambd,cl,zl)
    end

    # βₗ = zₗ₊₁ᴴ⋅zₗ₊₁/zₗᴴ⋅zₗ
    βl = dot(zl,zl)/ζl

    #pₗ = zₗ + βₗ*pₗ
    rmul!(pl,βl)
    BLAS.axpy!(one(T),zl,pl)

    solverInfo != nothing && storeInfo(solverInfo,norm(S*cl-u),norm(cl))
  end

  applyConstraints(cl, sparseTrafo, enforceReal, enforcePositive)
  
  return cl
end
