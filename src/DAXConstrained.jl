export DaxConstrained

mutable struct DaxConstrained{matT,T,Tsparse,U} <: AbstractLinearSolver
  S::matT
  u::Vector{T}
  λ::Float64
  B::Tsparse
  Bnorm²::Vector{Float64}
  denom::Vector{U}
  rowindex::Vector{Int64}
  zk::Vector{T}
  bk::Vector{T}
  bc::Vector{T}
  xl::Vector{T}
  yl::Vector{T}
  yc::Vector{T}
  δc::Vector{T}
  εw::Vector{T}
  τl::T
  αl::T
  weights::Vector{U}
  iterations::Int64
  iterationsInner::Int64
end

"""This function solves a constrained linear least squares problem using an algorithm proposed in [1].
Returns an approximate solution to Sᵀx = u s.t. Bx>=0 (each component >=0).

[1] Dax, A. On Row Relaxation Methods for Large Constrained Least Squares Problems. SIAM J. Sci. Comput. 14, 570–584 (1993).

### Input Arguments
* `S::AbstractMatrix{T}`: Problem Matrix S.
* `u::Vector{T}`: Righthandside of the linear equation.
* `B::AbstractMatrix{T}`: Transpose of Basistransformation if solving in dual space.

### Keyword/Optional Arguments

* `iterations::Int`: Number of Iterations of outer dax scheme.
* `iterationsInner::Int`: Number of Iterations of inner dax scheme.
* `λ::Float64`: The regularization parameter ɛ>0 influences the speed of convergence but not the solution.
* `weights::Bool`: Use weights in vector to weight equations. The larger the weight the more one 'trusts' a sqecific equation.
* `B`: Basistransformation if solving in dual space.
"""
function DaxConstrained(S, b=nothing; λ::Real=0.0
              , weights::Vector{R}=ones(Float64,size(S,1))
              , sparseTrafo=nothing
              , enforceReal::Bool=false
              , enforcePositive::Bool=false
              , iterations::Int=3
              , iterationsInner::Int=2
              , kargs...) where R <: Real

  T = typeof(real(S[1]))
  M,N = size(S)

  # setup denom and rowindex
  denom, rowindex = initkaczmarzconstraineddax(S,λ,weights)

  # set basis transformation
  sparseTrafo==nothing ? B=Matrix{T}(I, size(S,2), size(S,2)) : B=sparseTrafo
  Bnorm² = [rownorm²(B,i) for i=1:size(B,2)]

  if b != nothing
    u = b
  else
    u = zeros(eltype(S),M)
  end

  zk = zeros(eltype(S),N)
  bk = zeros(eltype(S),M)
  bc = zeros(T,size(B,2))
  xl = zeros(eltype(S),N)
  yl = zeros(eltype(S),M)
  yc = zeros(eltype(S),N)
  δc = zeros(eltype(S),N)
  εw = zeros(eltype(S),length(rowindex))
  τl = zero(eltype(S))
  αl = zero(eltype(S))

  return DaxConstrained(S,u,Float64(λ),B,Bnorm²,denom,rowindex,zk,bk,bc,xl,yl,yc,δc,εw,τl,αl
                  ,T.(weights),iterations,iterationsInner)
end

function init!(solver::DaxConstrained
              ; S::matT=solver.S
              , λ::Real=solver.λ
              , u::Vector{T}=eltype(S)[]
              , zk::Vector{T}=eltype(S)[]
              , weights::Vector{Float64}=solver.weights) where {matT,T}

  if S != solver.S
    denom, rowindex = initkaczmarzconstraineddax(S,λ,weights)
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
  solver.bc[:] .= zero(T)
  solver.xl[:] .= zero(T)
  solver.yl[:] .= zero(T)
  solver.yc[:] .= zero(T)
  solver.δc[:] .= zero(T)
  solver.αl = zero(T)        #temporary scalar
  solver.τl = zero(T)        #temporary scalar

  for i=1:length(solver.rowindex)
    j = solver.rowindex[i]
    solver.ɛw[i] = sqrt(solver.λ)/weights[j]
  end
end

function solve(solver::DaxConstrained, u::Vector{T}; λ::Real=solver.λ
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

function iterate(solver::DaxConstrained, iteration::Int=0)
  if done(solver,iteration) return nothing end

  # bk = u-S'*zk
  copyto!(solver.bk,solver.u)
  gemv!('N',-1.0,solver.S,solver.zk,1.0,solver.bk)

  # solve min ɛ|x|²+|W*A*x-W*bk|² with weightingmatrix W=diag(wᵢ), i=1,...,M.
  for l=1:solver.iterationsInner
    for i=1:length(solver.rowindex) # perform kaczmarz for all rows, which receive an update.
      j = solver.rowindex[i]
      solver.τl = dot_with_matrix_row(solver.S,solver.xl,j)
      solver.αl = solver.denom[i]*(solver.bk[j]-solver.τl-solver.ɛw[i]*solver.yl[j])
      kaczmarz_update!(solver.S,solver.xl,j,solver.αl)
      solver.yl[j] += solver.αl*solver.ɛw[i]
    end

    #Lent-Censor scheme for ensuring B(xl+zk) >= 0
    # copyto!(solver.δc,solver.xl)
    # BLAS.axpy!(1.0,solver.zk,solver.δc)
    # lmul!(solver.B, solver.δc)
    # lentcensormin!(solver.δc,solver.yc)
    #
    # BLAS.axpy!(1.0,solver.δc,solver.yc) # yc += δc
    #
    # δc = gemv('T',solver.B,solver.δc)
    # BLAS.axpy!(1.0,solver.solver.δc,solver.xl) # xl += Bᵀ*δc

    #Lent-Censor scheme for solving Bx >= 0
    # bc = xl + zk
    copyto!(solver.bc,solver.zk)
    BLAS.axpy!(1.0,solver.xl,solver.bc)

    for i=1:size(solver.B,2)
      δ = dot_with_matrix_row(solver.B,solver.bc,i)/solver.Bnorm²[i]
      δ = δ<solver.yc[i] ? -δ : -solver.yc[i]
      solver.yc[i] += δ
      kaczmarz_update!(solver.B,solver.xl,i,δ)  # update xl
      kaczmarz_update!(solver.B,solver.bc,i,δ)  # update bc
    end
  end
  BLAS.axpy!(1.0,solver.xl,solver.zk)  # zk += xl

  # reset xl and yl for next Kaczmarz run
  rmul!(solver.xl,0.0)
  rmul!(solver.yl,0.0)
  rmul!(solver.yc,0.0)

  return solver.bk, iteration+1

end

@inline done(solver::DaxConstrained,iteration::Int) = iteration>=solver.iterations

function lentcensormin!(x::Vector{T},y::Vector{T}) where {T<:Real}
  α = -one(T)
  @simd for i=1:length(x)
    @inbounds x[i]<=y[i] ? x[i] = α*x[i] : x[i] = α*y[i]
  end
end

"""This funtion saves the denominators to compute αl in denom and the rowindices,
  which lead to an update of cl in rowindex."""
function initkaczmarzconstraineddax(S::AbstractMatrix,ɛ::Number,weights::Vector)
  length(weights)==size(S,1) ? nothing : error("number of weights must equal number of equations")
  denom = Float64[]
  rowindex = Int64[]

  for i=1:size(S,1)
    s² = rownorm²(S,i)*weights[i]^2
    if s²>0
      push!(denom,weights[i]^2/(s²+ɛ))
      push!(rowindex,i)
    end
  end
  return denom, rowindex
end

"""
This funtion saves the denominators to compute αl in denom and the rowindices,
which lead to an update of cl in rowindex.
"""
function initkaczmarzconstraineddaxfft(S::AbstractMatrix,ɛ::Number,weights::Vector)
  length(weights)==size(S,1) ? nothing : error("number of weights must equal number of equations")
  denom = Float64[]
  rowindex = Int64[]

  for i=1:2:size(S,1)
    s²a = (rownorm²(S,i)+rownorm²(S,i+1))*weights[i]^2
    s²b = (rownorm²(S,i)+rownorm²(S,i+1))*weights[i+1]^2
    if s²a>0 && s²b>0
      push!(denom,weights[i]^2/(s²a+ɛ))
      push!(denom,weights[i+1]^2/(s²b+ɛ))
      push!(rowindex,i)
      push!(rowindex,i+1)
    end
  end
  return denom, rowindex
end
