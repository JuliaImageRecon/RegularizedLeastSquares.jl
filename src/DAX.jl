export daxrandkaczmarz, daxconstrained, daxconstrainedfft

mutable struct DaxKaczmarz <: AbstractLinearSolver
  A
  params
end

DaxKaczmarz(A; kargs...) = DaxKaczmarz(A,kargs)

function solve(solver::DaxKaczmarz, u::Vector)
  return daxrandkaczmarz(solver.A, u; solver.params... )
end


mutable struct DaxConstrained <: AbstractLinearSolver
  A
  params
end

DaxConstrained(A; kargs...) = DaxConstrained(A,kargs)


function solve(solver::DaxConstrained, u::Vector)
  return daxconstrained(solver.A, u; solver.params... )
end


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
function daxrandkaczmarz(S::AbstractMatrix, u::Vector;
 iterations=3, iterationsInner=2, λ=1e-1, solverInfo = nothing, weights = nothing, sparseTrafo = nothing, enforceReal = true, enforcePositive = true, kargs...)
  T = typeof(real(S[1]))
  λ = convert(T,λ)
  weights==nothing ? weights=ones(T,size(S,1)) : nothing #search for positive solution as default
  x = daxrand(S,u,iterations,iterationsInner,λ,solverInfo,weights)

  applyConstraints(x, sparseTrafo, enforceReal, enforcePositive)

  return x
end

""" This funtion saves the denominators to compute αl in denom and the rowindices,
which lead to an update of cl in rowindex.
"""
function daxrand(S::AbstractMatrix{T}, u::Vector, iterations::Int, iterationsInner::Int,
     λ::Number, solverInfo, weights::Vector) where T
  M = size(S,1)       #number of rows of system matrix
  N = size(S,2)       #number of cols of system matrix
  sumrowweights, denom, rowindex = initkaczmarzdax(S,λ,weights) #denom necessary to update αl, if rownorm ≠ 0. rowindex contains the indeces of nonzero rows. sumrowweight used for randomization of kaczmarz.

  zk = zeros(T,N)     #initialize zk
  bk = zeros(T,M)     #initialize uk
  xl = zeros(T,N)     #solution vector
  yl = zeros(T,M)     #residual vector
  αl = zero(T)        #temporary scalar
  τl = zero(T)        #temporary scalar

  ɛw = zeros(T,length(rowindex))        #temporary scalar
  for i=1:length(rowindex)
    j = rowindex[i]
    ɛw[i] = sqrt(λ)/weights[j]
  end

  reg = Regularization("L2", λ)

  for k=1:iterations #could be replaced by a while loop based on errorbound if smallest singular value of A is known
    # bk = u-S'*zk
    copyto!(bk,u)
    gemv!('N',-1.0,S,zk,1.0,bk)

    # solve min ɛ|x|²+|W*A*x-W*bk|² with weightingmatrix W=diag(wᵢ), i=1,...,M.
    for l=1:length(rowindex)*iterationsInner
      i::Int64 = getrandindex(sumrowweights,rand()*sumrowweights[end])  #choose row with propability proportional to its weight.
      j = rowindex[i]
      τl = dot_with_matrix_row(S,xl,j)
      αl = denom[i]*(bk[j]-τl-ɛw[i]*yl[j])
      kaczmarz_update!(S,xl,j,αl)
      yl[j] += αl*ɛw[i]
    end

    BLAS.axpy!(1.0,xl,zk)  # zk += xl
    # reset xl and yl for next Kaczmarz run
    rmul!(xl,0.0)
    rmul!(yl,0.0)

    # storeInfo(solverInfo,norm(bk),norm(zk))
    solverInfo != nothing && storeInfo(solverInfo,S,u,zk;reg=[reg],residual=bk)
  end
  return zk
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
function daxconstrained(S::AbstractMatrix{T}, u::Vector;
 iterations=3, iterationsInner=2, λ=1e-1, solverInfo=nothing,
 weights=nothing, sparseTrafo=nothing, kargs...) where {T<:Real}
  λ = convert(T,λ)
  sparseTrafo==nothing ? B=Matrix{T}(I, size(S,2), size(S,2)) : B=sparseTrafo #search for positive solution as default
  weights==nothing ? weights=ones(T,size(S,1)) : nothing #search for positive solution as default
  return daxcon(S, u, B, iterations, iterationsInner, λ, solverInfo, weights)
end

#function daxcon{T<:Real}(S::AbstractMatrix{T}, u::Vector, B, iterations::Int, iterationsInner::Int, λ::Number, solverInfo, weights::Vector{T})
#  error("This function still contains an error and needs fixing!")
#  #TODO fix bug. Constraints are implemented correctly, but algorithm does not converge to the correct solution.
#  M = size(S,2)       #number of equations
#  M%2 == 0 ? nothing : error("number of equations must be even")
#  N = size(S,1)       #number of unknowns
#  K = size(B,2)       #number of constraints
#  denom, rowindex = initkaczmarzconstraineddaxfft(S,λ,weights) #denom necessary to update αl, if rownorm ≠ 0. rowindex contains the indeces of nonzero rows.
#
#  zrk = zeros(T,N)     #initialize solution vector
#  zik = zeros(T,N)     #initialize solution vector
#  bk = zeros(T,M)     #initialize bk
#  xrl = zeros(T,N)     #intermediate solution
#  xil = zeros(T,N)     #intermediate solution
#  yl = zeros(T,M)     #dual of intermediate solution
#  yc = zeros(T,N*2)     #initialize part of dual var holding constraints
#  αl = zero(T)        #temporary scalar
#  τl = zero(T)        #temporary scalar
#  dk = zeros(ComplexF64,N)     #temporary vector
#  δ = zeros(T,N*2)
#
#  for k=1:iterations
#    # bk = u-S'*zk
#    copyto!(bk,u)
#    for i=1:2:M
#      bk[i] += -dot_with_matrix_row_simd(S,zrk,i)+dot_with_matrix_row_simd(S,zik,i+1)
#      bk[i+1] += -dot_with_matrix_row_simd(S,zik,i)-dot_with_matrix_row_simd(S,zrk,i+1)
#    end
#
##    # solve min ɛ|x|²+|W*A*x-W*bk|² with weightingmatrix W=diag(wᵢ), i=1,...,M.
#    for l=1:iterationsInner
#      for i=1:2:length(rowindex) # perform kaczmarz for all rows, which receive an update.
#        j = rowindex[i]
#        τl = dot_with_matrix_row_simd(S,xrl,j)-dot_with_matrix_row_simd(S,xil,j+1)
#        αl = denom[i]*(bk[j]-τl-sqrt(λ)/weights[j]*yl[j])
#        kaczmarz_update_simd!(S,xrl,j,αl)
#        kaczmarz_update_simd!(S,xil,j+1,-αl)
 #       yl[j] += αl*sqrt(λ)/weights[j]
#
#        τl = dot_with_matrix_row_simd(S,xil,j)+dot_with_matrix_row_simd(S,xrl,j+1)
#        αl = denom[i+1]*(bk[j+1]-τl-sqrt(λ)/weights[j+1]*yl[j+1])
#        kaczmarz_update_simd!(S,xrl,j+1,αl)
#        kaczmarz_update_simd!(S,xil,j,αl)
 #       yl[j+1] += αl*sqrt(λ)/weights[j+1]
#      end
#      #Lent-Censor scheme for solving Bx >= 0
#      for i=2:2:length(yc)
#        yc[i] = typemax(T)
#      end
#      #δ = -min(yc,B*(xl+zk))
#      for i=1:length(xrl)
#        dk[i] = xrl[i]+zrk[i] +im*(xil[i]+zik[i])
#      end
#      lmul!(B, dk)
#      δ = reinterpret(T,dk)
#      lentcensormin!(δ,yc)
#
#      BLAS.axpy!(1.0,δ,yc) # yc += δ
#
#      dk = reinterpret(ComplexF64,δ)
#      dk = gemv('T',B,dk)
#      δ = reinterpret(T,dk)
#      for i=2:2:length(δ)
#        xrl[div(i,2)] += δ[i-1]
#        xil[div(i,2)] += δ[i]
#      end
#    end
#
#    # zk += xl
#    BLAS.axpy!(1.0,xrl,zrk)
#    BLAS.axpy!(1.0,xil,zik)
#
#    # reset xl and yl for next Kaczmarz run
#    rmul!(xrl,0.0)
#    rmul!(xil,0.0)
#    rmul!(yl,0.0)
#    rmul!(yc,0.0)
#
#    storeInfo(solverInfo,norm(bk),norm(zrk)+norm(zik))
#  end
#  return zrk+im*zik
#end

function lentcensormin!(x::Vector{T},y::Vector{T}) where {T<:Real}
  α = -one(T)
  @simd for i=1:length(x)
    @inbounds x[i]<=y[i] ? x[i] = α*x[i] : x[i] = α*y[i]
  end
end

function daxcon(S::AbstractMatrix{T}, u::Vector, B, iterations::Int,
  iterationsInner::Int, λ::Number, solverInfo, weights::Vector{T}) where {T<:Real}
  M = size(S,1)       #number of equations
  N = size(S,2)       #number of unknowns
  denom, rowindex = initkaczmarzconstraineddax(S,λ,weights) #denom necessary to update αl, if rownorm ≠ 0. rowindex contains the indeces of nonzero rows.

  zk = zeros(T,N)     #initialize solution vector
  bk = zeros(T,M)     #initialize bk
  xl = zeros(T,N)     #intermediate solution
  yl = zeros(T,M)     #dual of intermediate solution
  yc = zeros(T,N)     #initialize part of dual var holding constraints
  δc = zeros(T,N)      #initialize temporary vector for Lent-Censor scheme
  αl = zero(T)        #temporary scalar
  τl = zero(T)        #temporary scalar

  ɛw = zeros(T,length(rowindex))        #temporary scalar
  for i=1:length(rowindex)
    j = rowindex[i]
    ɛw[i] = sqrt(λ)/weights[j]
  end

  reg = Regularization("L2", λ)

  for k=1:iterations
    # bk = u-S'*zk
    copyto!(bk,u)
    gemv!('N',-1.0,S,zk,1.0,bk)

    # solve min ɛ|x|²+|W*A*x-W*bk|² with weightingmatrix W=diag(wᵢ), i=1,...,M.
    for l=1:iterationsInner
      for i=1:length(rowindex) # perform kaczmarz for all rows, which receive an update.
        j = rowindex[i]
        τl = dot_with_matrix_row(S,xl,j)
        αl = denom[i]*(bk[j]-τl-ɛw[i]*yl[j])
        kaczmarz_update!(S,xl,j,αl)
        yl[j] += αl*ɛw[i]
      end

      #Lent-Censor scheme for ensuring B(xl+zk) >= 0
      copyto!(δc,xl)
      BLAS.axpy!(1.0,zk,δc)
      lmul!(B, δc)
      lentcensormin!(δc,yc)

      BLAS.axpy!(1.0,δc,yc) # yc += δc

      δc = gemv('T',B,δc)
      BLAS.axpy!(1.0,δc,xl) # xl += Bᵀ*δc
    end
    BLAS.axpy!(1.0,xl,zk)  # zk += xl

    # reset xl and yl for next Kaczmarz run
    rmul!(xl,0.0)
    rmul!(yl,0.0)
    rmul!(yc,0.0)

    # storeInfo(solverInfo,norm(bk),norm(zk))
    solverInfo != nothing && storeInfo(solverInfo,S,u,zk;reg=[reg],residual=bk)
  end
  return zk
end

function daxcon(S::AbstractMatrix{T}, u::Vector, B::AbstractMatrix, iterations::Int,
   iterationsInner::Int, λ::Number, solverInfo, weights::Vector{T}) where {T<:Real}
  M = size(S,1)       #number of equations
  N = size(S,2)       #number of unknowns
  K = size(B,2)       #number of constraints
  denom, rowindex = initkaczmarzconstraineddax(S,λ,weights) #denom necessary to update αl, if rownorm ≠ 0. rowindex contains the indeces of nonzero rows.
  Bnorm² = [rownorm²(B,i) for i=1:K]

  zk = zeros(T,N)     #initialize solution vector
  bk = zeros(T,M)     #initialize bk
  bc = zeros(T,K)     #initialize bc part of b holding the constraints
  xl = zeros(T,N)     #intermediate solution
  yl = zeros(T,M)     #dual of intermediate solution
  yc = zeros(T,K)     #initialize part of dual var holding constraints
  αl = zero(T)        #temporary scalar
  τl = zero(T)        #temporary scalar
  δ = zero(T)

  ɛw = zeros(T,length(rowindex))        #temporary scalar
  for i=1:length(rowindex)
    j = rowindex[i]
    ɛw[i] = sqrt(λ)/weights[j]
  end

  reg = Regularization("L2", λ)

  for k=1:iterations
    # bk = u-S'*zk
    copyto!(bk,u)
    gemv!('N',-1.0,S,zk,1.0,bk)

    # solve min ɛ|x|²+|W*A*x-W*bk|² with weightingmatrix W=diag(wᵢ), i=1,...,M.
    for l=1:iterationsInner
      for i=1:length(rowindex) # perform kaczmarz for all rows, which receive an update.
        j = rowindex[i]
        τl = dot_with_matrix_row(S,xl,j)
        αl = denom[i]*(bk[j]-τl-ɛw[i]*yl[j])
        kaczmarz_update!(S,xl,j,αl)
        yl[j] += αl*ɛw[i]
      end

      #Lent-Censor scheme for solving Bx >= 0
      # bc = xl + zk
      copyto!(bc,zk)
      BLAS.axpy!(1.0,xl,bc)

      for i=1:K
        δ = dot_with_matrix_row(B,bc,i)/Bnorm²[i]
        δ = δ<yc[i] ? -δ : -yc[i]
        yc[i] += δ
        kaczmarz_update!(B,xl,i,δ)  # update xl
        kaczmarz_update!(B,bc,i,δ)  # update bc
      end
    end
    BLAS.axpy!(1.0,xl,zk)  # zk += xl

    # reset xl and yl for next Kaczmarz run
    rmul!(xl,0.0)
    rmul!(yl,0.0)
    rmul!(yc,0.0)

    # storeInfo(solverInfo,norm(bk),norm(zk))
    solverInfo != nothing && storeInfo(solverInfo,S,u,zk;reg=[reg],residual=bk)
  end
  return zk
end
