export cgnr


type CGNR <: AbstractLinearSolver
  A
  params
end

CGNR(A; kargs...) = CGNR(A,kargs)

function init(solver::CGNR)
  nothing
end

function deinit(solver::CGNR)
  nothing
end

function solve(solver::CGNR, u::Vector)
  return cgnr(solver.A, u; solver.params... )
end



@doc "This funtion implements the cgnr algorithm." ->
function cgnr{T}(S::AbstractMatrix{T}, u::Vector{T};
iterations = 10, lambd::Real = 0.0, startVector = nothing, weights = nothing, enforceReal = false, enforcePositive = false, sparseTrafo = nothing, solverInfo = nothing, kargs... )
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
  copy!(rl,u)
  A_mul_B!(-one(T), S, cl, one(T), rl)
  #At_mul_B!(-one(T),S,cl,one(T),rl)
  #zₗ = Sᶜ*rₗ, where ᶜ denotes complex conjugation
  if weights != nothing
    xl = rl .* weights
    Ac_mul_B!(one(T), S, xl, zero(T), zl)
    #Aconj_mul_B!(one(T),S,xl,zero(T),zl)
  else
    Ac_mul_B!(one(T), S, rl, zero(T), zl)
    #Aconj_mul_B!(one(T),S,rl,zero(T),zl)
  end
  #pₗ = zₗ
  copy!(pl,zl)
  #start iteration
  p = Progress(iterations, 1, "CGNR Iteration...")
  for l=1:iterations
    #vₗ = Sᵗ*pₗ
    A_mul_B!(one(T), S, pl, zero(T), vl)
    #At_mul_B!(vl,S,pl)

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
      Ac_mul_B!(one(T), S, xl, zero(T), zl)
      #Aconj_mul_B!(one(T),S,xl,zero(T),zl)
    else
      Ac_mul_B!(one(T), S, rl, zero(T), zl)
      #Aconj_mul_B!(one(T),S,rl,zero(T),zl)
    end
    if lambd > 0
      BLAS.axpy!(-lambd,cl,zl)
    end

    # βₗ = zₗ₊₁ᴴ⋅zₗ₊₁/zₗᴴ⋅zₗ
    βl = dot(zl,zl)/ζl

    #pₗ = zₗ + βₗ*pₗ
    scale!(pl,βl)
    BLAS.axpy!(one(T),zl,pl)

    solverInfo != nothing && storeInfo(solverInfo,norm(S*cl-u),norm(cl))
    next!(p)
  end
  if sparseTrafo != nothing # This is a hack to allow constraints even when solving in a dual space
    A_mul_B!(sparseTrafo, cl)
  end
  if enforceReal #apply constraint: solution must be real
    enfReal!(cl)
  end
  if enforcePositive #apply constraint: solution must be real
    enfPos!(cl)
  end
  if sparseTrafo != nothing # backtransformation
    At_mul_B!(sparseTrafo, cl)
  end
  return cl
end

A_mul_B!{T}(α::Number,A::Matrix{T},x::AbstractArray{T,1},β::Number,y::AbstractArray{T,1}) = BLAS.gemv!('N', α, A, x, β, y)
At_mul_B!{T}(α::Number,A::Matrix{T},x::AbstractArray{T,1},β::Number,y::AbstractArray{T,1}) = BLAS.gemv!('T', α, A, x, β, y)
Ac_mul_B!{T}(α::Number,A::Matrix{T},x::AbstractArray{T,1},β::Number,y::AbstractArray{T,1}) = BLAS.gemv!('C', α, A, x, β, y)












# the idea of this solver is to dynamically increase the number of used rows in a cgnr solver
function tobitestcgnr{T}(A::AbstractMatrix{T}, b::Vector{T}; iterations=10, lambd=0, weights=nothing,
  shuff=false, enforceReal=true, enforcePositive=true, sparseTrafo=nothing, solverInfo=nothing )
  N = size(A,1)
  M = size(A,2)
  x = nothing

  for l=1:iterations
    numRows = min(M,l*10)
    w= weights==nothing ? nothing : weights[1:numRows]

    x = cgnr(A[:,1:numRows],b[1:numRows], iterations=3, lambd=lambd, weights=w)

    if solverInfo != nothing
      push!( solverInfo.xNorm, norm(x))
      push!( solverInfo.resNorm, norm(A.'x-b) )
    end
  end
  x
end










# original cgnr implementation has error.
function cgnrold{T}(A::AbstractMatrix{T}, b::Vector{T}; iterations=10, lambd=0, enforceReal=false, enforcePositive=false, sparseTrafo=nothing )
  M = size(A,2)
  N = size(A,1)
  NMMax = N > M ? N : M #defined but not used

  x = zeros(T, N)

  A = A.'

  r = A * x - b
  z = A' * r

  p = copy(z)

  progress = Progress(iterations, 1, "CGNR Iteration...")
  for j=1:iterations
    v = A*p

    alpha_tmp = norm(z)^2

    if lambd > 0
      alpha = alpha_tmp / ( norm(v)^2 + lambd*norm(p)^2 )
    else
      alpha = alpha_tmp / (norm(v)^2);
    end

    x += alpha*p



    ### constraints ###
    if sparseTrafo != nothing # This is a hack to allow constraints even when solving in a dual space
      x = sparseTrafo * x
    end

    # apply constraints
    constraint!(x, enforcePositive, enforceReal)

    if sparseTrafo != nothing # backtransformation
      x = sparseTrafo \ x
    end

    r = r - alpha*v
    z = A' * r


    if lambd > 0
      z = z -lambd*x
    end

    beta = norm(z)^2
    beta /= alpha_tmp

    p = z + beta*p

    next!(progress)
  end

  return x
end
