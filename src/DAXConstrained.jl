export DaxConstrained

mutable struct DaxConstrained{matT,T,Tsparse,U} <: AbstractRowActionSolver
  A::matT
  u::Vector{T}
  λ::Float64
  B::Tsparse
  Bnorm²::Vector{Float64}
  denom::Vector{U}
  rowindex::Vector{Int64}
  x::Vector{T}
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

"""
    DaxConstrained(A; b=nothing, λ=0, weights=ones(real(eltype(A)),size(A,1)), sparseTrafo=nothing, iterations=3, iterationsInner=2)

Creates an `DaxConstrained` object for the forward operator `A`. Solves a constrained linear least squares problem using an algorithm proposed in [1]. Returns an approximate solution to Sᵀx = u s.t. Bx>=0 (each component >=0).

[1] Dax, A. On Row Relaxation Methods for Large Constrained Least Squares Problems. SIAM J. Sci. Comput. 14, 570–584 (1993).

# Required Arguments
  * `A`                                                 - forward operator

# Optional Keyword Arguments
  * `b::AbstractMatrix`                                 - transpose of basistransformation if solving in dual space
  * `λ::Real`                                           - regularization parameter ɛ>0 influences the speed of convergence but not the solution.
  * `weights::AbstractVector`                           - the larger the weight the more one 'trusts' a sqecific equation
  * `iterations::Int`                                   - maximum number of (outer) ADMM iterations
  * `iterationsInner::Int`                              - max number of (inner) dax iterations

See also [`createLinearSolver`](@ref), [`solve`](@ref).
"""
function DaxConstrained(A
                      ; b=nothing
                      , λ::Real=0
                      , weights::AbstractVector=ones(real(eltype(A)),size(A,1))
                      , sparseTrafo=nothing
                      , iterations::Int=3
                      , iterationsInner::Int=2
                      )

  T = eltype(A)
  rT = real(eltype(A))
  M,N = size(A)

  # setup denom and rowindex
  denom, rowindex = initkaczmarzconstraineddax(A,λ,weights)

  # set basis transformation
  sparseTrafo === nothing ? B=Matrix{rT}(I, size(A,2), size(A,2)) : B=sparseTrafo
  Bnorm² = [rownorm²(B,i) for i=1:size(B,2)]

  if b !== nothing
    u = b
  else
    u = zeros(T,M)
  end

  x  = zeros(T,N)
  bk = zeros(T,M)
  bc = zeros(rT,size(B,2))
  xl = zeros(T,N)
  yl = zeros(T,M)
  yc = zeros(T,N)
  δc = zeros(T,N)
  εw = zeros(T,length(rowindex))
  τl = zero(T)
  αl = zero(T)

  return DaxConstrained(A,u,Float64(λ),B,Bnorm²,denom,rowindex,x,bk,bc,xl,yl,yc,δc,εw,τl,αl
                  ,rT.(weights),iterations,iterationsInner)
end

function init!(solver::DaxConstrained, b; x0 = 0)
  solver.u .= b
  solver.x .= x0

  solver.bk .= 0
  solver.bc .= 0
  solver.xl .= 0
  solver.yl .= 0
  solver.yc .= 0
  solver.δc .= 0
  solver.αl  = 0
  solver.τl  = 0

  for i=1:length(solver.rowindex)
    j = solver.rowindex[i]
    solver.ɛw[i] = sqrt(solver.λ)/solver.weights[j]
  end
end

solverconvergence(solver::DaxConstrained) = (; :residual = norm(solver.bk))

function iterate(solver::DaxConstrained, iteration::Int=0)
  if done(solver,iteration) return nothing end

  # bk = u-A'*x
  copyto!(solver.bk,solver.u)
  gemv!('N',-1.0,solver.A,solver.x,1.0,solver.bk)

  # solve min ɛ|x|²+|W*A*x-W*bk|² with weightingmatrix W=diag(wᵢ), i=1,...,M.
  for l=1:solver.iterationsInner
    for i=1:length(solver.rowindex) # perform kaczmarz for all rows, which receive an update.
      j = solver.rowindex[i]
      solver.τl = dot_with_matrix_row(solver.A,solver.xl,j)
      solver.αl = solver.denom[i]*(solver.bk[j]-solver.τl-solver.ɛw[i]*solver.yl[j])
      kaczmarz_update!(solver.A,solver.xl,j,solver.αl)
      solver.yl[j] += solver.αl*solver.ɛw[i]
    end

    #Lent-Censor scheme for ensuring B(xl+x) >= 0
    # copyto!(solver.δc,solver.xl)
    # BLAS.axpy!(1.0,solver.x,solver.δc)
    # lmul!(solver.B, solver.δc)
    # lentcensormin!(solver.δc,solver.yc)
    #
    # BLAS.axpy!(1.0,solver.δc,solver.yc) # yc += δc
    #
    # δc = gemv('T',solver.B,solver.δc)
    # BLAS.axpy!(1.0,solver.solver.δc,solver.xl) # xl += Bᵀ*δc

    #Lent-Censor scheme for solving Bx >= 0
    # bc = xl + x
    copyto!(solver.bc,solver.x)
    BLAS.axpy!(1.0,solver.xl,solver.bc)

    for i=1:size(solver.B,2)
      δ = dot_with_matrix_row(solver.B,solver.bc,i)/solver.Bnorm²[i]
      δ = δ<solver.yc[i] ? -δ : -solver.yc[i]
      solver.yc[i] += δ
      kaczmarz_update!(solver.B,solver.xl,i,δ)  # update xl
      kaczmarz_update!(solver.B,solver.bc,i,δ)  # update bc
    end
  end
  BLAS.axpy!(1.0,solver.xl,solver.x)  # x += xl

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

"""This function saves the denominators to compute αl in denom and the rowindices,
  which lead to an update of cl in rowindex."""
function initkaczmarzconstraineddax(A::AbstractMatrix,ɛ::Number,weights::Vector)
  length(weights)==size(A,1) ? nothing : error("number of weights must equal number of equations")
  denom = Float64[]
  rowindex = Int64[]

  for i=1:size(A,1)
    s² = rownorm²(A,i)*weights[i]^2
    if s²>0
      push!(denom,weights[i]^2/(s²+ɛ))
      push!(rowindex,i)
    end
  end
  return denom, rowindex
end

"""
This function saves the denominators to compute αl in denom and the rowindices,
which lead to an update of cl in rowindex.
"""
function initkaczmarzconstraineddaxfft(A::AbstractMatrix,ɛ::Number,weights::Vector)
  length(weights)==size(A,1) ? nothing : error("number of weights must equal number of equations")
  denom = Float64[]
  rowindex = Int64[]

  for i=1:2:size(A,1)
    s²a = (rownorm²(A,i)+rownorm²(A,i+1))*weights[i]^2
    s²b = (rownorm²(A,i)+rownorm²(A,i+1))*weights[i+1]^2
    if s²a>0 && s²b>0
      push!(denom,weights[i]^2/(s²a+ɛ))
      push!(denom,weights[i+1]^2/(s²b+ɛ))
      push!(rowindex,i)
      push!(rowindex,i+1)
    end
  end
  return denom, rowindex
end
