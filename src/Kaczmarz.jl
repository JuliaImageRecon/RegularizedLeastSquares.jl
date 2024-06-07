export kaczmarz
export Kaczmarz

mutable struct Kaczmarz{matT,R,T,U,RN,matAHA} <: AbstractRowActionSolver
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
  normalizeReg::AbstractRegularizationNormalization
  
  U_k::Vector{Int64} # CHANGE_KR
  greedy_randomized::Bool # CHANGE_KR
  theta::Union{Nothing,Float64}# CHANGE_KR
  e_k::T # CHANGE_KR
  norms::Vector{T}
  norm_size::Int64
  Fnorm::T
  r::Vector{T}
  B::matAHA
  i_k::Int64
  diff_vec_sq::Vector{T}
  diff_numb::T
  diff_denom::T
  r_probs::Vector{U}
end

"""
    Kaczmarz(A; reg = L2Regularization(0), normalizeReg = NoNormalization(), randomized=false, subMatrixFraction=0.15, shuffleRows=false, seed=1234, iterations=10, greedy_randomized=false, theta=Nothing)

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
  * `greedy_randomized::Bool`                             - use greedy randomized kaczmarz
  * `theta::Float64`                                      - set relaxation parameter theta

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
                , greedy_randomized::Bool = false # CHANGE_KR
                , theta::Union{Nothing,Float64}=nothing
                )

  T = real(eltype(A))

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

  # Tikhonov matrix is only valid with NoNormalization or SystemMatrixBasedNormalization
  if λ(L2) isa Vector && !(normalizeReg isa NoNormalization || normalizeReg isa SystemMatrixBasedNormalization)
    error("Tikhonov matrix for Kaczmarz is only valid with no or system matrix based normalization")
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

  M, N = size(A) # CHANGE_KR
  norms = zeros(eltype(A), M) # CHANGE_KR
  B = typeof(A)
  # setup denom and rowindex
  if greedy_randomized == true # CHANGE_KR
    A, denom, rowindex, norms = initkaczmarz(A, λ(L2), greedy_randomized)
    B = A * adjoint(A)  # CHANGE_KR
    # Calculate all denominators - B * 1/(||A||²)
    for x in 1:M
      B[:, x] = B[:, x] * denom[x]
    end
  else
  	A, denom, rowindex = initkaczmarz(A, λ(L2))
  end

  rowIndexCycle = collect(1:length(rowindex))
  probabilities = eltype(denom)[]
  if randomized
    probabilities = T.(rowProbabilities(A, rowindex))
  end

  subMatrixSize = round(Int, subMatrixFraction*M)

  u = zeros(eltype(A),M)
  x = zeros(eltype(A),N)
  vl = zeros(eltype(A),M)
  εw = zero(eltype(A))
  τl = zero(eltype(A))
  αl = zero(eltype(A))

  U_k = Int64[] # CHANGE_KR
  e_k = zero(eltype(A)) # CHANGE_KR
  norm_size = M
  Fnorm = eltype(A)(1.0 / norm(A, 2)^2)
  r = eltype(A)[]
  i_k = 0
  diff_vec_sq = zeros(eltype(A), M)
  diff_numb = zero(eltype(A))
  diff_denom = zero(eltype(A))
  r_probs = zeros(real(eltype(A)), M)
  
  return Kaczmarz(A, u, L2, other, denom, rowindex, rowIndexCycle, x, vl, εw, τl, αl,
                  randomized, subMatrixSize, probabilities, shuffleRows,
                  Int64(seed), iterations,
    normalizeReg, U_k, greedy_randomized, theta, e_k, norms, norm_size, Fnorm, r, B, i_k, diff_vec_sq, diff_numb, diff_denom, r_probs)
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
    # A must be unchanged, since we do not store the original SM
    _, solver.denom, solver.rowindex = initkaczmarz(solver.A, λ_)
    solver.rowIndexCycle = collect(1:length(rowindex))
    if solver.randomized
      solver.probabilities = T.(rowProbabilities(solver.A, rowindex))
    end
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
  if λ_ isa Vector
    solver.ɛw = 0
  else
    solver.ɛw = sqrt(λ_)
  end

  if solver.greedy_randomized
    solver.r = copy(b)
  end
end


function solversolution(solver::Kaczmarz{matT, RN}) where {matT, R<:L2Regularization{<:Vector}, RN <: Union{R, AbstractNestedRegularization{<:R}}}
  return solver.x .* (1 ./ sqrt.(λ(solver.L2)))
end
solversolution(solver::Kaczmarz) = solver.x
solverconvergence(solver::Kaczmarz) = (; :residual => norm(solver.vl))

function iterate(solver::Kaczmarz, iteration::Int=0)
  if done(solver,iteration) return nothing end

  if solver.randomized
    usedIndices = Int.(StatsBase.sample!(Random.GLOBAL_RNG, solver.rowIndexCycle, weights(solver.probabilities), zeros(solver.subMatrixSize), replace=false))
  # CHANGE_KR
  elseif solver.greedy_randomized
    calcDiff(solver)
    max = calcMax(solver)
    calcEk(solver, max, solver.theta)
    calcIndexSet(solver)
    calcProbSelection(solver)
    row = solver.rowindex[solver.i_k]
    iterate_row_index_greedy(solver, solver.A, row, solver.i_k)

    for r in solver.reg
      prox!(r, solver.x)
    end
    calcR(solver)
    return solver.vl, iteration + 1
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

  return solver.vl, iteration + 1
end

iterate_row_index(solver::Kaczmarz, A::AbstractLinearSolver, row, index) = iterate_row_index(solver, Matrix(A[row, :]), row, index) 
function iterate_row_index(solver::Kaczmarz, A, row, index)
  solver.τl = dot_with_matrix_row(A,solver.x,row)
  solver.αl = solver.denom[index]*(solver.u[row]-solver.τl-solver.ɛw*solver.vl[row])
  kaczmarz_update!(A,solver.x,row,solver.αl)
  solver.vl[row] += solver.αl*solver.ɛw
end

iterate_row_index_greedy(solver::Kaczmarz, A::AbstractLinearSolver, row, index) = iterate_row_index_greedy(solver, Matrix(A[row, :]), row, index)
function iterate_row_index_greedy(solver::Kaczmarz, A, row, index)
  solver.αl = solver.denom[index] * (solver.r[index] - solver.ɛw * solver.vl[row])
  kaczmarz_update!(A, solver.x, row, solver.αl)
  solver.vl[row] += solver.αl * solver.ɛw
end

@inline done(solver::Kaczmarz,iteration::Int) = iteration>=solver.iterations


"""
This function calculates the probabilities of the rows of the system matrix
"""

function rowProbabilities(A, rowindex)
  normA² = rownorm²(A, 1:size(A, 1))
  p = zeros(length(rowindex))
  for i=1:length(rowindex)
    j = rowindex[i]
    p[i] = rownorm²(A, j) / (normA²)
  end
  return p
end

# CHANGE_KR
# Calculate next useable index
function calcProbSelection(solver::Kaczmarz)
  r_denom = 1.0 / (norm(solver.r)^2)
  solver.r_probs = zeros(size(solver.r, 1))
  for i in solver.U_k
    if (i != 0)
      solver.r_probs[i] = (abs(solver.r[i])^2) * r_denom
  end
  end
  solver.i_k = sample(solver.U_k, ProbabilityWeights(solver.r_probs), 1, replace=false)[1]
end

### initkaczmarz ###

"""
    initkaczmarz(A::AbstractMatrix,λ)

This function saves the denominators to compute αl in denom and the rowindices,
which lead to an update of x in rowindex.
"""
function initkaczmarz(A,λ, greedy_randomized)
  T = real(eltype(A))
  denom = T[]
  rowindex = Int64[]
  norms = eltype(A)[] # CHANGE_KR
  for i = 1:size(A, 1)
    s² = rownorm²(A,i)
    if s²>0
      push!(denom,1/(s²+λ))
      push!(norms, s²) # CHANGE_KR
      push!(rowindex,i)
    end
  end
  return A, denom, rowindex, norms
end

function initkaczmarz(A,λ)
  T = real(eltype(A))
  denom = T[]
  rowindex = Int64[]
  for i = 1:size(A, 1)
    s² = rownorm²(A,i)
    if s²>0
      push!(denom,1/(s²+λ))
      push!(rowindex,i)
    end
  end
  return A, denom, rowindex
end

function initkaczmarz(A, λ::Vector)
  λ = real(eltype(A)).(λ)
  A = initikhonov(A, λ)
  return initkaczmarz(A, 0)
end

function calcMax(solver::Kaczmarz)
  max = zero(Float64)
  for i in 1:solver.norm_size
    new_max = convert(Float64, (solver.diff_vec_sq[i]) * (solver.denom[i]))
    if new_max > max
      max = new_max
    end
  end
  return max

end
function calcDiff(solver::Kaczmarz)
  solver.diff_vec_sq = map((x) -> abs(x)^2, solver.r)
  solver.diff_numb = norm(solver.r, 2)^2
  solver.diff_denom = 1.0 / solver.diff_numb
end

function calcEk(solver::Kaczmarz, max, theta::Nothing)
  solver.e_k = (0.5) * (((solver.diff_denom) * max) + solver.Fnorm)
end
function calcEk(solver::Kaczmarz, max, theta::Float64)
  solver.e_k = ((theta * ((solver.diff_denom) * max)) + ((1 - theta) * solver.Fnorm))
end
function calcIndexSet(solver::Kaczmarz)
  # Calculate e_K * || b - A*x_k ||²
  lower_bound_const = convert(Float64, solver.e_k * solver.diff_numb)
  solver.U_k = zeros(Int64, solver.norm_size)
  for i in 1:solver.norm_size
  # Calculate lower bound
    lower_bound = convert(Float64, lower_bound_const * solver.norms[i])
    # Check if index is in Set
    if convert(Float64, solver.diff_vec_sq[i]) >= lower_bound
      solver.U_k[i] = i
    end
  end
end


function calcR(solver::Kaczmarz)
  solver.r = solver.r - (solver.r[solver.i_k] * solver.B[:, solver.i_k])
end
# function calcRZero(solver::Kaczmarz, b::Vector{ComplexF64})
#   solver.r = copy(b)
# end

initikhonov(A, λ) = transpose((1 ./ sqrt.(λ)) .* transpose(A)) # optimize structure for row access
initikhonov(prod::ProdOp{Tc, WeightingOp{T}, matT}, λ) where {T, Tc<:Union{T, Complex{T}}, matT} = ProdOp(prod.A, initikhonov(prod.B, λ))
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
