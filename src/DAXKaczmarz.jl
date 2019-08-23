export DaxKaczmarz

mutable struct DaxKaczmarz{matT,T,U,Tsparse} <: AbstractLinearSolver
  S::matT
  u::Vector{T}
  λ::Float64
  denom::Vector{U}
  rowindex::Vector{Int64}
  sumrowweights::Vector{Float64}
  zk::Vector{T}
  bk::Vector{T}
  xl::Vector{T}
  yl::Vector{T}
  εw::Vector{T}
  τl::T
  αl::T
  weights::Vector{U}
  enforceReal::Bool
  enforcePositive::Bool
  sparseTrafo::Tsparse
  iterations::Int64
  iterationsInner::Int64
end

"""This function solves a unconstrained linear least squaares problem using an algorithm proposed in [1] combined with a randomized version of kaczmarz [2].
Returns an approximate solution to the linear leaast squares problem Sᵀx = u.

[1] Dax, A. On Row Relaxation Methods for Large Constrained Least Squares Problems. SIAM J. Sci. Comput. 14, 570–584 (1993).
[2] Strohmer, T. & Vershynin, R. A Randomized Kaczmarz Algorithm with Exponential Convergence. J. Fourier Anal. Appl. 15, 262–278 (2008).

### Input Arguments
* `S::AbstractMatrix{T}`: Problem Matrix S.
* `u::Vector{T}`: Righthandside of the linear equation.

### Keyword/Optional Arguments

* `iterations::Int`: Number of Iterations of outer dax scheme.
* `iterationsInner::Int`: Number of Iterations of inner dax scheme.
* `λ::Float64`: The regularization parameter ɛ>0 influences the speed of convergence but not the solution.
* `weights::Vector{T}`: Use weights in vector to weight equations. The larger the weight the more one 'trusts' a sqecific equation.
"""
function DaxKaczmarz(S, b=nothing; λ::Real=0.0
              , weights::Vector{R}=ones(Float64,size(S,1))
              , sparseTrafo=nothing
              , enforceReal::Bool=false
              , enforcePositive::Bool=false
              , iterations::Int=3
              , iterationsInner::Int=2
              , kargs...) where R <: Real

  # setup denom and rowindex
  sumrowweights, denom, rowindex = initkaczmarzdax(S,λ,weights)

  T = typeof(real(S[1]))
  M,N = size(S)
  if b != nothing
    u = b
  else
    u = zeros(eltype(S),M)
  end
  zk = zeros(eltype(S),N)
  bk = zeros(eltype(S),M)
  xl = zeros(eltype(S),N)
  yl = zeros(eltype(S),M)
  εw = zeros(eltype(S),length(rowindex))
  τl = zero(eltype(S))
  αl = zero(eltype(S))

  return DaxKaczmarz(S,u,Float64(λ),denom,rowindex,sumrowweights,zk,bk,xl,yl,εw,τl,αl
                  ,T.(weights),enforceReal,enforcePositive
                  ,sparseTrafo,iterations,iterationsInner)
end

function init!(solver::DaxKaczmarz
              ; S::matT=solver.S
              , λ::Real=solver.λ
              , u::Vector{T}=eltype(S)[]
              , zk::Vector{T}=eltype(S)[]
              , weights::Vector{Float64}=solver.weights) where {matT,T}

  if S != solver.S
    solver.sumrowweights, solver.denom, solver.rowindex = initkaczmarzdax(S,solver.λ,solver.weights)
  end
  solver.λ = Float64(λ)

  solver.u[:] .= u
  solver.weights=weights

  # start vector
  if isempty(zk)
    solver.zk[:] .= zeros(T,size(S,2))
  else
    solver.zk[:] .= x
  end

  solver.bk[:] .= zero(T)
  solver.xl[:] .= zero(T)
  solver.yl[:] .= zero(T)
  solver.αl = zero(T)        #temporary scalar
  solver.τl = zero(T)        #temporary scalar

  for i=1:length(solver.rowindex)
    j = solver.rowindex[i]
    solver.ɛw[i] = sqrt(solver.λ)/weights[j]
  end
end

function solve(solver::DaxKaczmarz, u::Vector{T}; λ::Real=solver.λ
                , S::matT=solver.S, startVector::Vector{T}=eltype(S)[]
                , weights::Vector=solver.weights
                , solverInfo=nothing, kargs...) where {T,matT}

  # initialize solver parameters
  init!(solver; S=S, λ=λ, u=u, zk=startVector, weights=weights)

  # log solver information
  solverInfo != nothing && storeInfo(solverInfo,solver.S,solver.u,solver.zk;residual=bk)

  # perform CGNR iterations
  for (iteration, item) = enumerate(solver)
    solverInfo != nothing && storeInfo(solverInfo,solver.S,solver.u,solver.zk;residual=bk)
  end

  return solver.zk
end

function iterate(solver::DaxKaczmarz, iteration::Int=0)
  if done(solver,iteration)
    applyConstraints(solver.zk, solver.sparseTrafo, solver.enforceReal, solver.enforcePositive)
    return nothing
  end

  copyto!(solver.bk, solver.u)
  gemv!('N',-1.0,solver.S,solver.zk,1.0,solver.bk)

  # solve min ɛ|x|²+|W*A*x-W*bk|² with weightingmatrix W=diag(wᵢ), i=1,...,M.
  for l=1:length(solver.rowindex)*solver.iterationsInner
    i::Int64 = getrandindex(solver.sumrowweights,rand()*solver.sumrowweights[end])  #choose row with propability proportional to its weight.
    j = solver.rowindex[i]
    solver.τl = dot_with_matrix_row(solver.S,solver.xl,j)
    solver.αl = solver.denom[i]*(solver.bk[j]-solver.τl-solver.ɛw[i]*solver.yl[j])
    kaczmarz_update!(solver.S,solver.xl,j,solver.αl)
    solver.yl[j] += solver.αl*solver.ɛw[i]
  end

  BLAS.axpy!(1.0,solver.xl,solver.zk)  # zk += xl
  # reset xl and yl for next Kaczmarz run
  rmul!(solver.xl,0.0)
  rmul!(solver.yl,0.0)

  return solver.bk, iteration+1

end

@inline done(solver::DaxKaczmarz,iteration::Int) = iteration>=solver.iterations

"""This funtion saves the denominators to compute αl in denom and the rowindices,
  which lead to an update of cl in rowindex.
"""
function initkaczmarzdax(S::AbstractMatrix, ɛ, weights::Vector)
  length(weights)==size(S,1) ? nothing : error("number of weights must equal number of equations")
  denom = Float64[]
  sumrowweights = Float64[]
  rowindex = Int64[]

  push!(sumrowweights,0.0)
  for i=1:size(S,1)
    s² = rownorm²(S,i)*weights[i]^2
    if s²>0
      push!(denom,weights[i]^2/(s²+ɛ))
      push!(sumrowweights,s²+ɛ+sumrowweights[end])
      push!(rowindex,i)
    end
  end
  return sumrowweights, denom, rowindex
end

"This function returns the next index for the randomized Kaczmarz algorithm."
function getrandindex(v::AbstractVector, x)
    lo::Int64 = 0
    hi::Int64 = length(v)+1
    while lo < hi-1
        m = (lo+hi)>>>1
        x<v[m] ? hi = m : lo = m
    end
    return lo
end
