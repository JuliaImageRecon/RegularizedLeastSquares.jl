export kaczmarz
export Kaczmarz

mutable struct Kaczmarz{matT,T,U,Tsparse} <: AbstractLinearSolver
  S::matT
  u::Vector{T}
  reg::Vector{<:AbstractRegularization}
  denom::Vector{U}
  rowindex::Vector{Int64}
  rowIndexCycle::Vector{Int64}
  cl::Vector{T}
  vl::Vector{T}
  εw::Vector{T}
  τl::T
  αl::T
  weights::Vector{U}
  enforceReal::Bool
  enforcePositive::Bool
  shuffleRows::Bool
  seed::Int64
  sparseTrafo::Tsparse
  iterations::Int64
  regMatrix::Union{Nothing,Vector{U}} # Tikhonov regularization matrix
  normalizeReg::AbstractRegularizationNormalization
  normalizedReg::Vector{<:AbstractRegularization}
end

"""
  Kaczmarz(S, b=nothing; λ::Real=0.0, reg = Regularization("L2", λ)
              , weights::Vector{R}=ones(Float64,size(S,1))
              , sparseTrafo=nothing
              , enforceReal::Bool=false
              , enforcePositive::Bool=false
              , shuffleRows::Bool=false
              , seed::Int=1234
              , iterations::Int64=10
              , kargs...) where R <: Real

creates a Kaczmarz object

# Arguments
* `S`                                             - system matrix
* `b=nothing`                                     - measurement
* (`λ::Real=0.0`)                                 - regularization parameter
* (`reg=Regularization("L2", λ)`)                 - Regularization object
* (` weights::Vector{R}=ones(Float64,size(S,1))`) - weights for the data term
* (`sparseTrafo=nothing`)                         - sparsifying transformation
* (`enforceReal::Bool=false`)                     - constrain the solution to be real
* (`enforcePositive::Bool=false`)                 - constrain the solution to have positive real part
* (`shuffleRows::Bool=false`)                     - randomize Kacmarz algorithm
* (`seed::Int=1234`)                              - seed for randomized algorithm
* (iterations::Int64=10)                          - number of iterations
"""
function Kaczmarz(S; b=nothing, reg = L2Regularization(0.0)
              , weights=nothing
              , sparseTrafo=nothing
              , enforceReal::Bool=false
              , enforcePositive::Bool=false
              , shuffleRows::Bool=false
              , seed::Int=1234
              , iterations::Int64=10
              , regMatrix=nothing
              , normalizeReg::AbstractRegularizationNormalization = NoNormalization()
              , kargs...)

  T = real(eltype(S))

  if !(reg isa L2Regularization || (reg isa Vector && reg[1] isa L2Regularization))
    error("Kaczmarz only supports L2 regularizer as first regularization term")
  end
  reg = vec(reg)

  # Apply Tikhonov regularization matrix 
  if regMatrix != nothing
    regMatrix = T.(regMatrix) # make sure regMatrix has the same element type as S 
    S = transpose(1 ./ sqrt.(regMatrix)) .* S # apply Tikhonov regularization to system matrix
  end

  # make sure weights are not empty
  w = (weights!=nothing ? weights : ones(T,size(S,1)))

  # normalization parameters
  normalizedReg = normalize(Kaczmarz, normalizeReg, reg, S, nothing)

  # setup denom and rowindex
  denom, rowindex = initkaczmarz(S, λ(normalizedReg[1]), w)
  rowIndexCycle=collect(1:length(rowindex))

  M,N = size(S)
  if b != nothing
    u = b
  else
    u = zeros(eltype(S),M)
  end
  cl = zeros(eltype(S),N)
  vl = zeros(eltype(S),M)
  εw = zeros(eltype(S),length(rowindex))
  τl = zero(eltype(S))
  αl = zero(eltype(S))

  return Kaczmarz(S,u,reg,denom,rowindex,rowIndexCycle,cl,vl,εw,τl,αl
                  ,T.(w),enforceReal,enforcePositive,shuffleRows
                  ,Int64(seed),sparseTrafo,iterations, regMatrix, 
                  normalizeReg, normalizedReg)
end

"""
  init!(solver::Kaczmarz
              ; S::matT=solver.S
              , u::Vector{T}=T[]
              , cl::Vector{T}=T[]
              , shuffleRows=solver.shuffleRows) where {T,matT}

(re-) initializes the CGNR iterator
"""
function init!(solver::Kaczmarz
              ; S::matT=solver.S
              , u::Vector{T}=T[]
              , cl::Vector{T}=T[]
              , weights::Vector{R}=solver.weights
              , shuffleRows=solver.shuffleRows) where {T,matT,R}
              
  solver.normalizedReg = normalize(solver, solver.normalizeReg, solver.normalizedReg, S, u)
  
  λ_ = λ(solver.normalizedReg[1])
  if S != solver.S
    solver.denom, solver.rowindex = initkaczmarz(S, λ_, weights)
    solver.rowIndexCycle = collect(1:length(solver.rowindex))
  end

  if shuffleRows
    Random.seed!(solver.seed)
    shuffle!(solver.rowIndexCycle)
  end
  solver.u[:] .= u
  solver.weights=weights

  # start vector
  if isempty(cl)
    solver.cl[:] .= zero(T)
  else
    solver.cl[:] .= cl
  end
  solver.vl[:] .= zero(T)

  for i=1:length(solver.rowindex)
    j = solver.rowindex[i]
    solver.ɛw[i] = sqrt(λ_) / weights[j]
  end

end

"""
  solve(solver::Kaczmarz, u::Vector{T};
                S::matT=solver.S, startVector::Vector{T}=eltype(S)[]
                , weights::Vector=solver.weights, shuffleRows::Bool=false
                , solverInfo=nothing, kargs...) where {T,matT}

solves Tikhonov-regularized inverse problem using Kaczmarz algorithm.

# Arguments
* `solver::Kaczmarz  - the solver containing both system matrix and regularizer
* `u::Vector`        - data vector
* (`S::matT=solver.S`)                  - operator for the data-term of the problem
* (`startVector::Vector{T}=T[]`)        - initial guess for the solution
* (`weights::Vector{T}=solver.weights`) - weights for the data term
* (`shuffleRows::Bool=false`)           - randomize Kacmarz algorithm
* (`solverInfo=nothing`)                - solverInfo for logging

when a `SolverInfo` objects is passed, the residuals are stored in `solverInfo.convMeas`.
"""
function solve(solver::Kaczmarz, u::Vector{T};
                S::matT=solver.S, startVector::Vector{T}=eltype(S)[]
                , weights::Vector=solver.weights, shuffleRows::Bool=false
                , solverInfo=nothing, kargs...) where {T,matT}

  # initialize solver parameters
  init!(solver; S=S, u=u, cl=startVector, weights=weights, shuffleRows=shuffleRows)

  # log solver information
  solverInfo != nothing && storeInfo(solverInfo,solver.cl,norm(solver.vl))

  # perform Kaczmarz iterations
  for item in solver
    solverInfo != nothing && storeInfo(solverInfo,solver.cl,norm(solver.vl))
  end

  # backtransformation of solution with Tikhonov matrix
  if solver.regMatrix != nothing
    solver.cl = solver.cl .* (1 ./ sqrt.(solver.regMatrix))
  end

  return solver.cl
end

function iterate(solver::Kaczmarz, iteration::Int=0)
  if done(solver,iteration) return nothing end

  for i in solver.rowIndexCycle
    j = solver.rowindex[i]
    solver.τl = dot_with_matrix_row(solver.S,solver.cl,j)
    solver.αl = solver.denom[i]*(solver.u[j]-solver.τl-solver.ɛw[i]*solver.vl[j])
    kaczmarz_update!(solver.S,solver.cl,j,solver.αl)
    solver.vl[j] += solver.αl*solver.ɛw[i]
  end

  # We skip the L2 regularizer, since it has already been applied
  for r in solver.reg[2:end]
    prox!(r, solver.cl)
  end

  return solver.vl, iteration+1

end

@inline done(solver::Kaczmarz,iteration::Int) = iteration>=solver.iterations

### initkaczmarz ###

"""
    initkaczmarz(S::AbstractMatrix,λ,weights::Vector)

This funtion saves the denominators to compute αl in denom and the rowindices,
which lead to an update of cl in rowindex.
"""
function initkaczmarz(S::AbstractMatrix,λ,weights::Vector)
  T = typeof(real(S[1]))
  denom = T[]
  rowindex = Int64[]

  for i=1:size(S,1)
    s² = rownorm²(S,i)*weights[i]^2
    if s²>0
      push!(denom,weights[i]^2/(s²+λ))
      push!(rowindex,i)
    end
  end
  denom, rowindex
end

### kaczmarz_update! ###

"""
    kaczmarz_update!(A::DenseMatrix{T}, x::Vector, k::Integer, beta) where T

This funtion updates x during the kaczmarz algorithm for dense matrices.
"""
function kaczmarz_update!(A::DenseMatrix{T}, x::Vector, k::Integer, beta) where T
  @simd for n=1:size(A,2)
    @inbounds x[n] += beta*conj(A[k,n])
  end
end

"""
    kaczmarz_update!(B::Transpose{T,S}, x::Vector,
                     k::Integer, beta) where {T,S<:DenseMatrix}

This funtion updates x during the kaczmarz algorithm for dense matrices.
"""
function kaczmarz_update!(B::Transpose{T,S}, x::Vector,
			  k::Integer, beta) where {T,S<:DenseMatrix}
  A = B.parent
  @inbounds @simd for n=1:size(A,1)
      x[n] += beta*conj(A[n,k])
  end
end

# kaczmarz_update! with manual simd optimization
for (T,W, WS,shufflevectorMask,vσ) in [(Float32,:WF32,:WF32S,:shufflevectorMaskF32,:vσF32),(Float64,:WF64,:WF64S,:shufflevectorMaskF64,:vσF64)]
    eval(quote
        const $WS = VectorizationBase.pick_vector_width($T)
        const $W = Int(VectorizationBase.pick_vector_width($T))
        const $shufflevectorMask = Val(ntuple(k -> iseven(k-1) ? k : k-2, $W))
        const $vσ = Vec(ntuple(k -> (-1f0)^(k+1),$W)...)
        function kaczmarz_update!(A::Transpose{Complex{$T},S}, b::Vector{Complex{$T}}, k::Integer, beta::Complex{$T}) where {S<:DenseMatrix}
            b = reinterpret($T,b)
            A = reinterpret($T,A.parent)

            N = length(b)
            Nrep, Nrem = divrem(N,4*$W) # main loop
            Mrep, Mrem = divrem(Nrem,$W) # last iterations
            idx = MM{$W}(1)
            iOffset = 4*$W

            vβr = vbroadcast($WS, beta.re) * $vσ # vector containing (βᵣ,-βᵣ,βᵣ,-βᵣ,...)
            vβi = vbroadcast($WS, beta.im) # vector containing (βᵢ,βᵢ,βᵢ,βᵢ,...)

            GC.@preserve b A begin # protect A and y from GC
                vptrA = stridedpointer(A)
                vptrb = stridedpointer(b)
                for _ = 1:Nrep
                    Base.Cartesian.@nexprs 4 i -> vb_i = vload(vptrb, ($W*(i-1) + idx,))
                    Base.Cartesian.@nexprs 4 i -> va_i = vload(vptrA, ($W*(i-1) + idx,k))
                    Base.Cartesian.@nexprs 4 i -> begin
                        vb_i = muladd(va_i, vβr, vb_i)
                        va_i = shufflevector(va_i, $shufflevectorMask)
                        vb_i = muladd(va_i, vβi, vb_i)
                    	vstore!(vptrb, vb_i, ($W*(i-1) + idx,))
                    end
                    idx += iOffset
                end

                for _ = 1:Mrep
	            vb = vload(vptrb, (idx,))
	            va = vload(vptrA, (idx,k))
                    vb = muladd(va, vβr, vb)
                    va = shufflevector(va, $shufflevectorMask)
                    vb = muladd(va, vβi, vb)
		            vstore!(vptrb, vb, (idx,))
                    idx += $W
                end

                if Mrem!=0
                    vloadMask = VectorizationBase.mask($T, Mrem)
                    vb = vload(vptrb, (idx,), vloadMask)
                    va = vload(vptrA, (idx,k), vloadMask)
                    vb = muladd(va, vβr, vb)
                    va = shufflevector(va, $shufflevectorMask)
                    vb = muladd(va, vβi, vb)
                    vstore!(vptrb, vb, (idx,), vloadMask)
                end
            end # GC.@preserve
        end
    end)
end

#=
@doc "This funtion updates x during the kaczmarz algorithm for dense matrices." ->
function kaczmarz_update!{T}(A::Matrix{T}, x::Vector{T}, k::Integer, beta::T)
  BLAS.axpy!(length(x), beta, pointer(A,sub2ind(size(A),1,k)), 1, pointer(x), 1)
end
=#

"""
    kaczmarz_update!(B::Transpose{T,S}, x::Vector,
                          k::Integer, beta) where {T,S<:SparseMatrixCSC}

This funtion updates x during the kaczmarz algorithm for sparse matrices.
"""
function kaczmarz_update!(B::Transpose{T,S}, x::Vector,
                          k::Integer, beta) where {T,S<:SparseMatrixCSC}
  A = B.parent
  N = A.colptr[k+1]-A.colptr[k]
  for n=A.colptr[k]:N-1+A.colptr[k]
    @inbounds x[A.rowval[n]] += beta*conj(A.nzval[n])
  end
end
