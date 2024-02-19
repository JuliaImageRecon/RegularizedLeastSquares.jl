export kaczmarz
export Kaczmarz

mutable struct Kaczmarz{matT,T,U,R,RN} <: AbstractRowActionSolver
  A::matT
  u::Vector{T}
  L2::R
  reg::Vector{RN}
  denom::Vector{U}
  rowindex::Vector{Int64}
  rowIndexCycle::Vector{Int64}
  x::Vector{T}
  vl::Vector{T}
  εw::T
  τl::T
  αl::T
  randomized::Bool
  subMatrixSize::Int64
  probabilities::Vector{U}
  shuffleRows::Bool
  seed::Int64
  iterations::Int64
  regMatrix::Union{Nothing,Vector{U}} # Tikhonov regularization matrix
  normalizeReg::AbstractRegularizationNormalization
end

"""
    Kaczmarz(A; reg = L2Regularization(0), normalizeReg = NoNormalization(), weights=nothing, randomized=false, subMatrixFraction=0.15, shuffleRows=false, seed=1234, iterations=10, regMatrix=nothing)

Creates a Kaczmarz object for the forward operator `A`.

# Required Arguments
  * `A`                                                 - forward operator

# Optional Keyword Arguments
  * `reg::AbstractParameterizedRegularization`          - regularization term
  * `normalizeReg::AbstractRegularizationNormalization` - regularization normalization scheme; options are `NoNormalization()`, `MeasurementBasedNormalization()`, `SystemMatrixBasedNormalization()`
  * `randomized::Bool`                                    - randomize Kacmarz algorithm
  * `subMatrixFraction::Real`                             - fraction of rows used in randomized Kaczmarz algorithm
  * `shuffleRows::Bool`                                   - randomize Kacmarz algorithm
  * `seed::Int`                                           - seed for randomized algorithm
  * `iterations::Int`                                     - number of iterations

See also [`createLinearSolver`](@ref), [`solve!`](@ref).
"""
function Kaczmarz(A
                ; reg = L2Regularization(zero(real(eltype(A))))
                , normalizeReg::AbstractRegularizationNormalization = NoNormalization()
                , randomized::Bool = false
                , subMatrixFraction::Real = 0.15
                , shuffleRows::Bool = false
                , seed::Int = 1234
                , iterations::Int = 10
                , regMatrix = nothing
                )

  T = real(eltype(A))

  # Apply Tikhonov regularization matrix
  if regMatrix !== nothing
    regMatrix = T.(regMatrix) # make sure regMatrix has the same element type as A
    A = transpose(1 ./ sqrt.(regMatrix)) .* A # apply Tikhonov regularization to system matrix
  end

  # Prepare regularization terms
  reg = isa(reg, AbstractVector) ? reg : [reg]
  reg = normalize(Kaczmarz, normalizeReg, reg, A, nothing)
  idx = findsink(L2Regularization, reg)
  if isnothing(idx)
    L2 = L2Regularization(zero(T))
  else
    L2 = reg[idx]
    deleteat!(reg, idx)
  end

  indices = findsinks(AbstractProjectionRegularization, reg)
  other = AbstractRegularization[reg[i] for i in indices]
  deleteat!(reg, indices)
  if length(reg) == 1
    push!(other, reg[1])
  elseif length(reg) > 1
    error("Kaczmarz does not allow for more than one additional regularization term, found $(length(reg))")
  end
  other = identity.(other)

  # setup denom and rowindex
  denom, rowindex = initkaczmarz(A, λ(L2))
  rowIndexCycle = collect(1:length(rowindex))
  probabilities = T.(rowProbabilities(A, rowindex))

  M,N = size(A)
  subMatrixSize = round(Int, subMatrixFraction*M)

  u  = zeros(eltype(A),M)
  x = zeros(eltype(A),N)
  vl = zeros(eltype(A),M)
  εw = zero(eltype(A))
  τl = zero(eltype(A))
  αl = zero(eltype(A))

  return Kaczmarz(A, u, L2, other, denom, rowindex, rowIndexCycle, x, vl, εw, τl, αl,
                  randomized, subMatrixSize, probabilities, shuffleRows,
                  Int64(seed), iterations, regMatrix,
                  normalizeReg)
end

"""
  init!(solver::Kaczmarz, b; x0 = 0)

(re-) initializes the Kacmarz iterator
"""
function init!(solver::Kaczmarz, b; x0 = 0)
  λ_prev = λ(solver.L2)
  solver.L2  = normalize(solver, solver.normalizeReg, solver.L2,  solver.A, b)
  solver.reg = normalize(solver, solver.normalizeReg, solver.reg, solver.A, b)

  λ_ = λ(solver.L2)

  # λ changed => recompute denoms
  if λ_ != λ_prev
    solver.denom, solver.rowindex = initkaczmarz(solver.A, λ_)
    solver.rowIndexCycle = collect(1:length(rowindex))
    solver.probabilities = T.(rowProbabilities(solver.A, rowindex))  
  end

  if solver.shuffleRows || solver.randomized
    Random.seed!(solver.seed)
  end
  if solver.shuffleRows
    shuffle!(solver.rowIndexCycle)
  end

  # start vector
  solver.x .= x0
  solver.vl .= 0

  solver.u .= b
  solver.ɛw = sqrt(λ_)
end


function solversolution(solver::Kaczmarz)
  # backtransformation of solution with Tikhonov matrix
  if solver.regMatrix !== nothing
    return solver.x .* (1 ./ sqrt.(solver.regMatrix))
  end
  return solver.x
end
solverconvergence(solver::Kaczmarz) = (; :residual => norm(solver.vl))

function iterate(solver::Kaczmarz, iteration::Int=0)
  if done(solver,iteration) return nothing end

  if solver.randomized
    usedIndices = Int.(StatsBase.sample!(Random.GLOBAL_RNG, solver.rowIndexCycle, weights(solver.probabilities), zeros(solver.subMatrixSize), replace=false))
  else
    usedIndices = solver.rowIndexCycle
  end

  for i in usedIndices
    row = solver.rowindex[i]
    iterate_row_index(solver, solver.A, row, i)
  end

  for r in solver.reg
    prox!(r, solver.x)
  end

  return solver.vl, iteration+1
end

iterate_row_index(solver::Kaczmarz, A::AbstractLinearSolver, row, index) = iterate_row_index(solver, Matrix(A[row, :]), row, index) 
function iterate_row_index(solver::Kaczmarz, A, row, index)
  solver.τl = dot_with_matrix_row(A,solver.x,row)
  solver.αl = solver.denom[index]*(solver.u[row]-solver.τl-solver.ɛw*solver.vl[row])
  kaczmarz_update!(A,solver.x,row,solver.αl)
  solver.vl[row] += solver.αl*solver.ɛw
end

@inline done(solver::Kaczmarz,iteration::Int) = iteration>=solver.iterations


"""
This function calculates the probabilities of the rows of the system matrix
"""

function rowProbabilities(A::AbstractMatrix, rowindex)
  M,N = size(A)
  normS = norm(A)^2
  p = zeros(length(rowindex))
  for i=1:length(rowindex)
    j = rowindex[i]
    p[i] = (norm(A[j,:]))^2 / (normS)
  end

  return p
end

rowProbabilities(A::AbstractLinearOperator, rowindex) = rowProbabilities(Matrix(A[rowindex, :]), 1:length(rowindex))
rowProbabilities(A::ProdOp{T, <:WeightingOp, matT}, rowindex) where {T, matT} = rowProbabilities(A.B, rowindex)

### initkaczmarz ###

"""
    initkaczmarz(A::AbstractMatrix,λ,weights::Vector)

This function saves the denominators to compute αl in denom and the rowindices,
which lead to an update of x in rowindex.
"""
function initkaczmarz(A,λ)
  T = real(eltype(A))
  denom = T[]
  rowindex = Int64[]

  for i in 1:size(A, 1)
    s² = rownorm²(A,i)
    if s²>0
      push!(denom,1/(s²+λ))
      push!(rowindex,i)
    end
  end
  denom, rowindex
end

### kaczmarz_update! ###

"""
    kaczmarz_update!(A::DenseMatrix{T}, x::Vector, k::Integer, beta) where T

This function updates x during the kaczmarz algorithm for dense matrices.
"""
function kaczmarz_update!(A::DenseMatrix{T}, x::Vector, k::Integer, beta) where T
  @simd for n=1:size(A,2)
    @inbounds x[n] += beta*conj(A[k,n])
  end
end

"""
    kaczmarz_update!(B::Transpose{T,S}, x::Vector,
                     k::Integer, beta) where {T,S<:DenseMatrix}

This function updates x during the kaczmarz algorithm for dense matrices.
"""
function kaczmarz_update!(B::Transpose{T,S}, x::Vector,
			  k::Integer, beta) where {T,S<:DenseMatrix}
  A = B.parent
  @inbounds @simd for n=1:size(A,1)
      x[n] += beta*conj(A[n,k])
  end
end

function kaczmarz_update!(prod::ProdOp{Tc, WeightingOp{T}, matT}, x::Vector, k, beta) where {T, Tc<:Union{T, Complex{T}}, matT}
  weight = prod.A.weights[k]
  kaczmarz_update!(prod.B, x, k, weight*beta) # only for real weights
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
@doc "This function updates x during the kaczmarz algorithm for dense matrices." ->
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
