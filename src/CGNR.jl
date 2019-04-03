export cgnr


mutable struct CGNR <: AbstractLinearSolver
  A
  reg::Regularization
  params
end

function CGNR(A; λ = 0.0, reg = Regularization("L2", λ), kargs...)
  if (reg.prox!) != (proxL2!)
    @error "CGNR only supports L2 regularizer"
  end
  return CGNR(A,reg,kargs)
end

function solve(solver::CGNR, u::Vector)
  return cgnr(solver.A, u; λ=solver.reg.λ, solver.params... )
end


"""
This funtion implements the cgnr algorithm.
"""
function cgnr(S, u::Vector{T};
iterations = 10, λ::Real = 0.0, startVector = nothing, weights = nothing,
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

  reg = Regularization("L2", λ)

  #pre iteration
  #rl = u - Sᵗ*cl
  copyto!(rl,u)
  ###gemv!('N',-one(T), S, cl, one(T), rl)
  rl[:] = u - S*cl
  #zl = Sᶜ*rl, where ᶜ denotes complex conjugation
  if weights != nothing
    xl = rl .* weights
    ## gemv!('C',one(T), S, xl, zero(T), zl)
    zl[:] = adjoint(S)*xl
  else
    ## gemv!('C',one(T), S, rl, zero(T), zl)
    zl[:] = adjoint(S)*rl
  end
  #pl = zl
  copyto!(pl,zl)
  #start iteration
  for l=1:min(iterations,size(S,2))
    #vl = Sᵗ*pl
    ##gemv!('N',one(T), S, pl, zero(T), vl)
    vl[:] = S*pl

    # αl = zlᴴ⋅zl/(vlᴴ⋅vl+λ*plᴴ⋅pl)
    ζl = norm(zl)^2
    normvl = weights == nothing ? dot(vl,vl) : dot(vl,weights.*vl)
    λ > 0 ? αl = ζl/(normvl+λ*norm(pl)^2) : αl = ζl/normvl

    #cl += αl*pl
    BLAS.axpy!(αl,pl,cl)

    # if solverInfo != nothing
    #   push!( solverInfo.xNorm, norm(cl))
    #   push!( solverInfo.resNorm, norm(rl) )
    # end

    #rl += -αl*vl
    BLAS.axpy!(-αl,vl,rl)

    #zl = Sᶜ*rl-λ*cl
    if weights != nothing
      xl = rl .* weights
      ##gemv!('C',one(T), S, xl, zero(T), zl)
      zl[:] = adjoint(S)*xl
    else
      ##gemv!('C',one(T), S, rl, zero(T), zl)
      zl[:] = adjoint(S)*rl
    end
    if λ > 0
      BLAS.axpy!(-λ,cl,zl)
    end

    # βl = zl₊₁ᴴ⋅zl₊₁/zlᴴ⋅zl
    βl = dot(zl,zl)/ζl

    #pl = zl + βl*pl
    rmul!(pl,βl)
    BLAS.axpy!(one(T),zl,pl)

    solverInfo != nothing && storeInfo(solverInfo,S,u,cl;reg=[reg])
  end

  applyConstraints(cl, sparseTrafo, enforceReal, enforcePositive)

  return cl
end
