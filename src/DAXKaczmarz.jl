export DaxKaczmarz

mutable struct DaxKaczmarz{matT,N,T,U} <: AbstractRowActionSolver
  A::matT
  shape::NTuple{N, Int64}
  u::Vector{T}
  reg::Vector{<:AbstractRegularization}
  λ::Float64
  denom::Vector{U}
  rowindex::Vector{Int64}
  sumrowweights::Vector{Float64}
  x::Vector{T}
  bk::Vector{T}
  xl::Vector{T}
  yl::Vector{T}
  εw::Vector{T}
  τl::T
  αl::T
  weights::Vector{U}
  iterations::Int64
  iterationsInner::Int64
end

"""
    DaxKaczmarz(A; b=nothing, λ=0, weights=ones(real(eltype(A)),size(A,1)), sparseTrafo=nothing, enforceReal=false, enforcePositive=false, iterations=3, iterationsInner=2)

Creates an `DaxKaczmarz` object for the forward operator `A`. Solves a unconstrained linear least squares problem using an algorithm proposed in [1] combined with a randomized version of kaczmarz [2]. Returns an approximate solution to the linear least squares problem Sᵀx = u.

[1] Dax, A. On Row Relaxation Methods for Large Constrained Least Squares Problems. SIAM J. Sci. Comput. 14, 570–584 (1993).
[2] Strohmer, T. & Vershynin, R. A Randomized Kaczmarz Algorithm with Exponential Convergence. J. Fourier Anal. Appl. 15, 262–278 (2008).

# Required Arguments
  * `A`                                                 - forward operator

# Optional Keyword Arguments
  * `b::AbstractMatrix`                                 - transpose of basistransformation if solving in dual space
  * `λ::Real`                                           - regularization parameter ɛ>0 influences the speed of convergence but not the solution.
  * `sparseTrafo`                                       - TODO
  * `enforceReal::Bool`                                 - TODO
  * `enforcePositive::Bool`                             - TODO
  * `iterations::Int`                                   - maximum number of (outer) ADMM iterations
  * `iterationsInner::Int`                              - max number of (inner) dax iterations

See also [`createLinearSolver`](@ref), [`solve!`](@ref).
"""
function DaxKaczmarz(A
                   ; b=nothing
                   , λ::Real=0
                   , weights::Vector=ones(real(eltype(A)),size(A,1))
                   , sparseTrafo=nothing
                   , enforceReal::Bool=false
                   , enforcePositive::Bool=false
                   , iterations::Int=3
                   , iterationsInner::Int=2
                   , shape = (size(A, 1),)
                   )

  # setup denom and rowindex
  sumrowweights, denom, rowindex = initkaczmarzdax(A,λ,weights)

  T = typeof(real(A[1]))
  M,N = size(A)
  if b != nothing
    u = b
  else
    u = zeros(eltype(A),M)
  end
  x  = zeros(eltype(A),N)
  bk = zeros(eltype(A),M)
  xl = zeros(eltype(A),N)
  yl = zeros(eltype(A),M)
  εw = zeros(eltype(A),length(rowindex))
  τl = zero(eltype(A))
  αl = zero(eltype(A))

  reg = AbstractRegularization[]
  if enforcePositive && enforceReal
    push!(reg, PositiveRegularization())
  elseif enforceReal
    push!(reg, RealRegularization())
  end
  if !isempty(reg) && !isnothing(sparseTrafo)
    reg = map(r -> TransformedRegularization(r, sparseTrafo), reg)
  end
  return DaxKaczmarz(A,shape,u,reg, Float64(λ), denom,rowindex,sumrowweights,x,bk,xl,yl,εw,τl,αl
                  ,T.(weights) ,iterations,iterationsInner)
end

function init!(solver::DaxKaczmarz, b; x0 = 0)
  solver.u .= b
  solver.x .= x0

  solver.bk .= 0
  solver.xl .= 0
  solver.yl .= 0
  solver.αl  = 0
  solver.τl  = 0

  for i=1:length(solver.rowindex)
    j = solver.rowindex[i]
    solver.ɛw[i] = sqrt(solver.λ)/solver.weights[j]
  end
end

function iterate(solver::DaxKaczmarz, iteration::Int=0)
  if done(solver,iteration)
    for r in solver.reg
      prox!(r, reshape(solver.x, solver.shape))
    end
    return nothing
  end

  copyto!(solver.bk, solver.u)
  gemv!('N',-1.0,solver.A,solver.x,1.0,solver.bk)

  # solve min ɛ|x|²+|W*A*x-W*bk|² with weightingmatrix W=diag(wᵢ), i=1,...,M.
  for l=1:length(solver.rowindex)*solver.iterationsInner
    i::Int64 = getrandindex(solver.sumrowweights,rand()*solver.sumrowweights[end])  #choose row with propability proportional to its weight.
    j = solver.rowindex[i]
    solver.τl = dot_with_matrix_row(solver.A,solver.xl,j)
    solver.αl = solver.denom[i]*(solver.bk[j]-solver.τl-solver.ɛw[i]*solver.yl[j])
    kaczmarz_update!(solver.A,solver.xl,j,solver.αl)
    solver.yl[j] += solver.αl*solver.ɛw[i]
  end

  BLAS.axpy!(1.0,solver.xl,solver.x)  # x += xl
  # reset xl and yl for next Kaczmarz run
  rmul!(solver.xl,0.0)
  rmul!(solver.yl,0.0)

  return solver.bk, iteration+1

end

@inline done(solver::DaxKaczmarz,iteration::Int) = iteration>=solver.iterations

"""This function saves the denominators to compute αl in denom and the rowindices,
  which lead to an update of cl in rowindex.
"""
function initkaczmarzdax(A::AbstractMatrix, ɛ, weights::Vector)
  length(weights)==size(A,1) ? nothing : error("number of weights must equal number of equations")
  denom = Float64[]
  sumrowweights = Float64[]
  rowindex = Int64[]

  push!(sumrowweights,0.0)
  for i=1:size(A,1)
    s² = rownorm²(A,i)*weights[i]^2
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
