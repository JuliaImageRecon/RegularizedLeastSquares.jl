export kaczmarz

mutable struct Kaczmarz{matT,T,U,Tsparse} <: AbstractLinearSolver
  S::matT
  u::Vector{T}
  reg::Vector{Regularization}
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
  constraintMask::Union{Nothing,Vector{Bool}}
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
function Kaczmarz(S; b=nothing, λ=[0.0], reg = nothing
              , regName = ["L2"]
              , weights=nothing
              , sparseTrafo=nothing
              , enforceReal::Bool=false
              , enforcePositive::Bool=false
              , shuffleRows::Bool=false
              , seed::Int=1234
              , iterations::Int64=10
              , constraintMask=nothing
              , kargs...)

  if typeof(λ) <: Number
    λ = [λ]
  end

  if reg == nothing
    reg = Regularization(regName, λ; kargs...)
  end

  if regName[1] != "L2"
    error("Kaczmarz only supports L2 regularizer")
  end

  T = real(eltype(S))

  # make sure weights are not empty
  w = (weights!=nothing ? weights : ones(T,size(S,1)))

  # setup denom and rowindex
  denom, rowindex = initkaczmarz(S, λ[1], w)
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
                  ,Int64(seed),sparseTrafo,iterations, constraintMask)
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

  if S != solver.S
    solver.denom, solver.rowindex = initkaczmarz(S,solver.reg.λ,weights)
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
    solver.ɛw[i] = sqrt(solver.reg[1].λ) / weights[j]
  end
end

"""
  solve(solver::Kaczmarz, u::Vector{T};
                S::matT=solver.S, startVector::Vector{T}=eltype(S)[]
                , weights::Vector=solver.weights, shuffleRows::Bool=false
                , solverInfo=nothing, kargs...) where {T,matT}

solves Thikhonov-regularized inverse problem using Kaczmarz algorithm.

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
  solverInfo != nothing && storeInfo(solverInfo,solver.cl,solver.vl)

  # perform Kaczmarz iterations
  for item in solver
    solverInfo != nothing && storeInfo(solverInfo,solver.cl,solver.vl)
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

  # invoke constraints
  applyConstraints(solver.cl, solver.sparseTrafo,
                              solver.enforceReal,
                              solver.enforcePositive,
                              solver.constraintMask)

  if length(solver.reg) > 1
    # We skip the L2 regularizer, since it has already been applied
    prox!(solver.cl, solver.reg[2:end])
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
for (T,W,shufflevectorMask,vσ) in [(Float32,:WF32,:shufflevectorMaskF32,:vσF32),(Float64,:WF64,:shufflevectorMaskF64,:vσF64)]
    eval(quote
        const $W = VectorizationBase.pick_vector_width($T)
        const $shufflevectorMask = Val(ntuple(k -> iseven(k-1) ? k : k-2, $W))
        const $vσ = Vec{$W,$T}(ntuple(k -> (-1f0)^(k+1),$W))
        function kaczmarz_update!(A::Transpose{Complex{$T},S}, b::Vector{Complex{$T}}, k::Integer, beta::Complex{$T}) where {S<:DenseMatrix}
            b = reinterpret($T,b)
            A = reinterpret($T,A.parent)

            N = length(b)
            Nrep, Nrem = divrem(N,4*$W) # main loop
            Mrep, Mrem = divrem(Nrem,$W) # last iterations
            ib = _MM{$W}(0)
            ia = _MM{$W}((k-1)*stride(A,2))
            iOffset = 4*$W

            vβr = vmul(vbroadcast(Vec{$W,$T}, beta.re),$vσ) # vector containing (βᵣ,-βᵣ,βᵣ,-βᵣ,...)
            vβi = vbroadcast(Vec{$W,$T}, beta.im) # vector containing (βᵢ,βᵢ,βᵢ,βᵢ,...)

            GC.@preserve b A begin # protect A and y from GC
                vptrA = stridedpointer(A)
                vptrb = stridedpointer(b)
                for _ = 1:Nrep
			Base.Cartesian.@nexprs 4 i -> vb_i = vload(vptrb, ($W*(i-1) + ib,))
			Base.Cartesian.@nexprs 4 i -> va_i = vload(vptrA, ($W*(i-1) + ia,))
                    Base.Cartesian.@nexprs 4 i -> begin
                        vb_i = vmuladd(va_i, vβr, vb_i)
                        va_i = shufflevector(va_i, $shufflevectorMask)
                        vb_i = vmuladd(va_i, vβi, vb_i)
			vstore!(vptrb, vb_i, ($W*(i-1) + ib,))
                    end
                    ib += iOffset
                    ia += iOffset
                end

                for _ = 1:Mrep
			vb = vload(vptrb, (ib,))
			va = vload(vptrA, (ia,))
                    vb = vmuladd(va, vβr, vb)
                    va = shufflevector(va, $shufflevectorMask)
                    vb = vmuladd(va, vβi, vb)
		    vstore!(vptrb, vb, (ib,))
                    ib += $W
                    ia += $W
                end

                if Mrem!=0
                    vloadMask = VectorizationBase.mask($T, Mrem)
		    vb = vload(vptrb, (ib,), vloadMask)
		    va = vload(vptrA, (ia,), vloadMask)
                    vb = vmuladd(va, vβr, vb)
                    va = shufflevector(va, $shufflevectorMask)
                    vb = vmuladd(va, vβi, vb)
		    vstore!(vptrb, vb, (ib,), vloadMask)
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

### kaczmarz ###

"""
    kaczmarz(S, u::Vector; kargs...)

This funtion implements the kaczmarz algorithm.

# Keyword/Optional Arguments
* `λ::Float64`: The regularization parameter, relative to the matrix trace
* `iterations::Int`: Number of iterations of the iterative solver
* `solver::AbstractString`: Algorithm used to solve the imaging equation (currently "kaczmarz" or "cgnr")
* `normWeights::Bool`: Enable row normalization (true/false)
* `sparseTrafo::AbstractString`: Enable sparseTrafo if set to "DCT-IV" or "FFT".
* `shuff::Bool`: Enable shuffeling of rows during iterations in the kaczmarz algorithm.
* `enforceReal::Bool`: Enable projection of solution on real plane during iteration.
* `enforcePositive::Bool`: Enable projection of solution onto positive halfplane during iteration.
"""
