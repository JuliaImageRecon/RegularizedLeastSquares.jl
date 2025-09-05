export kaczmarz
export Kaczmarz


mutable struct Kaczmarz{matT,R,U,RN} <: AbstractRowActionSolver
  A::matT
  L2::R
  reg::Vector{RN}
  denom::Vector{U}
  rowindex::Vector{Int64}
  rowIndexCycle::Vector{Int64}
  randomized::Bool
  subMatrixSize::Int64
  probabilities::Vector{U}
  shuffleRows::Bool
  seed::Int64
  normalizeReg::AbstractRegularizationNormalization
  iterations::Int64
  state::AbstractSolverState{<:Kaczmarz}
end

mutable struct KaczmarzState{T, vecT <: AbstractArray{T}} <: AbstractSolverState{Kaczmarz}
  u::vecT
  x::vecT
  vl::vecT
  εw::T
  τl::T
  αl::T
  iteration::Int64
end

mutable struct GreedyKaczmarzState{T, vecT <: AbstractArray{T}, matAHA <: AbstractMatrix{T}, U, vecU <: AbstractArray{U}, vecI <: AbstractArray{Int64}} <: AbstractSolverState{Kaczmarz}
  u::vecT
  x::vecT
  vl::vecT
  εw::T
  τl::T
  αl::T
  iteration::Int64
  U_k::vecI
  theta::Union{Nothing,Float64}
  e_k::U
  norms::vecU
  norm_size::Int64
  Fnorm::T
  r::vecT
  B::matAHA
  diff_vec_sq::vecU
  diff_numb::U
  diff_denom::U
  r_probs::vecU
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
  ; reg=L2Regularization(zero(real(eltype(A)))), normalizeReg::AbstractRegularizationNormalization=NoNormalization(), randomized::Bool=false, subMatrixFraction::Real=0.15, shuffleRows::Bool=false, seed::Int=1234, iterations::Int=10, greedy_randomized::Bool=false, theta::Union{Nothing,Float64}=nothing
)

  T = real(eltype(A))

  # Prepare regularization terms
  reg = copy(isa(reg, AbstractVector) ? reg : [reg])
  reg = normalize(Kaczmarz, normalizeReg, reg, A, nothing)
  idx = findsink(L2Regularization, reg)
  if isnothing(idx)
    L2 = L2Regularization(zero(T))
  else
    L2 = reg[idx]
    deleteat!(reg, idx)
  end

  # Tikhonov matrix is only valid with NoNormalization or SystemMatrixBasedNormalization
  if λ(L2) isa AbstractVector && !(normalizeReg isa NoNormalization || normalizeReg isa SystemMatrixBasedNormalization)
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

  M, N = size(A)
  B = typeof(A)
  norms = zeros(eltype(T), M)
  # setup denom and rowindex
  if greedy_randomized
    B = (A * adjoint(A)) + (λ(L2) * I) # Potentially do this lazily/matrix free?
    A, denom, rowindex, norms = initgreedykaczmarz(A, λ(L2))
    for x in 1:M
      B[:, x] = (B[:, x]) * denom[x]
    end
  else
    A, denom, rowindex = initkaczmarz(A, λ(L2))
  end

  rowIndexCycle = collect(1:length(rowindex))
  probabilities = eltype(denom)[]
  if randomized
    probabilities = T.(rowProbabilities(A, rowindex))
  end

  subMatrixSize = round(Int, subMatrixFraction * M)

  u = zeros(eltype(A), M)
  x = zeros(eltype(A), N)
  vl = zeros(eltype(A), M)
  εw = zero(eltype(A))
  τl = zero(eltype(A))
  αl = zero(eltype(A))

  if !greedy_randomized
    state = KaczmarzState(u, x, vl, εw, τl, αl, 0)
  else
    e_k = zero(eltype(denom))
    norm_size = M
    U_k = zeros(Int64, norm_size)
    Fnorm = eltype(A)(1.0 / ((norm(A, 2)^2) + λ(L2)))
    r_probs = zeros(eltype(T), norm_size) # Inhalt auf 0 setzen statt allokieren solver.r_probs .= zero(eltype(solver.r_probs))
    r = zeros(eltype(A), norm_size)
    i_k = 0
    diff_vec_sq = zeros(eltype(T), M)
    diff_numb = zero(eltype(T))
    diff_denom = zero(eltype(T))  
    state = GreedyKaczmarzState(u, x, vl, εw, τl, αl, 0, U_k, theta, e_k, norms, norm_size, Fnorm, r, B, diff_vec_sq, diff_numb, diff_denom, r_probs)
  end

  return Kaczmarz(A, L2, other, denom, rowindex, rowIndexCycle,
                  randomized, subMatrixSize, probabilities, shuffleRows,
                  Int64(seed), normalizeReg, iterations, state)
end


function init!(solver::Kaczmarz, state::KaczmarzState{T, vecT}, b::otherT; kwargs...) where {T, vecT, otherT <: AbstractVector}
  u = similar(b, size(state.u)...)
  x = similar(b, size(state.x)...)
  vl = similar(b, size(state.vl)...)

  state = KaczmarzState(u, x, vl, state.εw, state.τl, state.αl, state.iteration)
  solver.state = state
  init!(solver, state, b; kwargs...)
end


"""
  init!(solver::Kaczmarz, b; x0 = 0)

(re-) initializes the Kacmarz iterator
"""
function init!(solver::Kaczmarz, state::KaczmarzState{T, vecT}, b::vecT; x0 = 0) where {T, vecT <: AbstractVector}
  λ_prev = λ(solver.L2)
  solver.L2 = normalize(solver, solver.normalizeReg, solver.L2, solver.A, b)
  solver.reg = normalize(solver, solver.normalizeReg, solver.reg, solver.A, b)

  λ_ = λ(solver.L2)

  # λ changed => recompute denoms
  if λ_ != λ_prev
    # A must be unchanged, since we do not store the original SM
    _, solver.denom, solver.rowindex = initkaczmarz(solver.A, λ_)
    solver.rowIndexCycle = collect(1:length(solver.rowindex))
    if solver.randomized
      solver.probabilities = T.(rowProbabilities(solver.A, solver.rowindex))
    end
  end

  if solver.shuffleRows || solver.randomized
    Random.seed!(solver.seed)
  end
  if solver.shuffleRows
    shuffle!(solver.rowIndexCycle)
  end

  # start vector
  state.x .= x0
  state.vl .= 0

  state.u .= b
  if λ_ isa AbstractVector
    state.ɛw = one(T)
  else
    state.ɛw = sqrt(λ_)
  end
  state.iteration = 0
end

function init!(solver::Kaczmarz, state::GreedyKaczmarzState{T, vecT}, b::vecT; x0 = 0) where {T, vecT <: AbstractVector}
  λ_prev = λ(solver.L2)
  solver.L2 = normalize(solver, solver.normalizeReg, solver.L2, solver.A, b)
  solver.reg = normalize(solver, solver.normalizeReg, solver.reg, solver.A, b)

  λ_ = λ(solver.L2)

  # λ changed
  if λ_ != λ_prev
    error("Measurement based regularization normalization is not supported for Greedy Kaczmarz")
  end

  # start vector
  state.x .= x0
  state.vl .= 0

  state.u .= b
  if λ_ isa AbstractVector
    state.ɛw = one(T)
  else
    state.ɛw = sqrt(λ_)
  end
  state.iteration = 0

  if x0 == 0
    state.r = copy(b)
  else
    # If x0 is not set to zero Vector
    state.r = copy(b)
    Ax = A * state.x
    Ax_reg = Ax .- state.εw
    state.r = state.r - Ax_reg
  end

end


function solversolution(solver::Kaczmarz{matT, RN}) where {matT, R<:L2Regularization{<:AbstractVector}, RN <: Union{R, AbstractNestedRegularization{<:R}}}
  return solversolution(solver.state) .* (1 ./ sqrt.(λ(solver.L2)))
end
solversolution(solver::Kaczmarz) = solversolution(solver.state)
solverconvergence(state::KaczmarzState) = (; :residual => norm(state.vl))

function iterate(solver::Kaczmarz, state::KaczmarzState)
  if done(solver,state) return nothing end

  if solver.randomized
    usedIndices = Int.(StatsBase.sample!(Random.GLOBAL_RNG, solver.rowIndexCycle, weights(solver.probabilities), zeros(solver.subMatrixSize), replace=false))
  else
    usedIndices = solver.rowIndexCycle
  end
  for i in usedIndices
    row = solver.rowindex[i]
    iterate_row_index(solver, state, solver.A, row, i)
  end

  for r in solver.reg
    prox!(r, state.x)
  end

  state.iteration += 1
  return state.x, state
end

function iterate(solver::Kaczmarz, state::GreedyKaczmarzState)
  if done(solver,state) return nothing end

  unused = 1
  for i in Base.OneTo(solver.subMatrixSize)
    iterate_row_index(solver, state, solver.A, unused, i)
  end

  for r in solver.reg
    prox!(r, state.x)
  end

  # Update residuals after proximal map application
  state.r .= state.u - (solver.A * state.x) - (state.ɛw * state.vl)

  state.iteration += 1
  return state.x, state
end

function iterate_row_index(solver::Kaczmarz, state::KaczmarzState, A, row, index)
  state.τl = dot_with_matrix_row(A,state.x,row)
  state.αl = solver.denom[index]*(state.u[row]-state.τl-state.ɛw*state.vl[row])
  kaczmarz_update!(A,state.x,row,state.αl)
  state.vl[row] += state.αl*state.ɛw
end

function iterate_row_index(solver::Kaczmarz, state::GreedyKaczmarzState, A, _, index)
  row = prepareGreedyKaczmarz(solver, state)
  state.αl = solver.denom[index] * (state.r[index])
  state.r .-= ((state.r[row]) .* (view(state.B, :, row)))
  state.τl = dot_with_matrix_row(A,state.x,row)
  state.αl = solver.denom[index]*(state.u[row]-state.τl-state.ɛw*state.vl[row])
  kaczmarz_update!(A,state.x,row,state.αl)
  state.vl[row] += state.αl*state.ɛw
end


@inline done(solver::Kaczmarz,state::AbstractSolverState{Kaczmarz}) = state.iteration>=solver.iterations


"""
This function calculates the probabilities of the rows of the system matrix
"""

function rowProbabilities(A, rowindex)
  normA² = rownorm²(A, 1:size(A, 1))
  p = zeros(length(rowindex))
  for i = 1:length(rowindex)
    j = rowindex[i]
    p[i] = rownorm²(A, j) / (normA²)
  end
  return p
end




### initkaczmarz ###
function initgreedykaczmarz(A, λ)
  T = real(eltype(A))
  denom = T[]
  norms = T[]
  rowindex = Int64[]
  for i = 1:size(A, 1)
    s² = rownorm²(A, i)
    if s² > 0
      norm = (s² + λ)
      push!(norms, norm)
      push!(denom, 1.0 / norm)
      push!(rowindex, i)
    end
  end
  return A, denom, rowindex, norms
end


"""
    initkaczmarz(A::AbstractMatrix,λ)

This function saves the denominators to compute αl in denom and the rowindices,
which lead to an update of x in rowindex.
"""
function initkaczmarz(A, λ)
  T = real(eltype(A))
  denom = T[]
  rowindex = Int64[]
  for i = 1:size(A, 1)
    s² = rownorm²(A, i)
    if s² > 0
      push!(denom, 1.0 / (s² + λ))
      push!(rowindex, i)
    end
  end
  return A, denom, rowindex
end
function initkaczmarz(A, λ::AbstractVector)
  # Instead of ||Ax - b||² + λ||x||² we solve ||Ax - b||² + λ||Lx||² where L is a diagonal matrix
  # See Per Christian Hansen: Rank-Deficient and Discrete Ill-Posed Problems, Chapter 2.3 Transformation to Standard Form
  # -> ||Âc - u||² + λ||c||² with Â = AL⁻¹, c = Lx
  # We put this into the standard extended system of equation with λ = 1
  # In the end we need to multiply the solution with L⁻¹ 
  λ = real(eltype(A)).(λ)
  A = initikhonov(A, λ)
  return initkaczmarz(A, one(eltype(λ)))
end

# A * inv(λ), specialised for diagm(λ)
initikhonov(A, λ::AbstractVector) =  transpose((1 ./ sqrt.(λ)) .* transpose(A)) # optimize structure for row access
initikhonov(prod::ProdOp{Tc, <:WeightingOp, matT}, λ) where {T, Tc<:Union{T, Complex{T}}, matT} = ProdOp(prod.A, initikhonov(prod.B, λ))

function prepareGreedyKaczmarz(solver::Kaczmarz, state::GreedyKaczmarzState)
  # Compute e_k
  state.r_probs .= abs2.(state.r)
  state.diff_numb = sum(state.r_probs)
  state.diff_denom = 1.0 / state.diff_numb
  # Inplace maximum(diff_vec_sq .* denom)
  max = maximum(Broadcast.instantiate(Broadcast.broadcasted(*, state.r_probs, solver.denom)))
  state.e_k = calcEk(state, max, state.theta)

  # Determine the index set of positive integers
  lower_bound_const = state.e_k * state.diff_numb
  # zero zeros below lower_bound_const and accumulate valid ones 
  r_sum = zero(eltype(state.r_probs))
  for i in eachindex(state.r_probs)
    val = state.r_probs[i]
    tmp = ifelse(val >= lower_bound_const * state.norms[i], val, zero(eltype(state.r_probs)))
    state.r_probs[i] = tmp
    r_sum += tmp
  end

  # Calculate the probability of selection
  r_denom = 1.0 / r_sum
  state.r_probs .*= r_denom

  # Select row
  return sample(Random.GLOBAL_RNG, ProbabilityWeights(state.r_probs, r_sum * r_denom))
end


function calcEk(state::GreedyKaczmarzState, max, theta::Nothing)
  return (0.5) * (((state.diff_denom) * max) + state.Fnorm)
end
function calcEk(state::GreedyKaczmarzState, max, theta::Float64)
  return ((theta * ((state.diff_denom) * max)) + ((1 - theta) * state.Fnorm))
end

### kaczmarz_update! ###

"""
    kaczmarz_update!(A::DenseMatrix{T}, x::Vector, k::Integer, beta) where T

This function updates x during the kaczmarz algorithm for dense matrices.
"""
function kaczmarz_update!(A::DenseMatrix{T}, x::Vector, k::Integer, beta) where {T}
  @simd for n = 1:size(A, 2)
    @inbounds x[n] += beta * conj(A[k, n])
  end
end

"""
    kaczmarz_update!(B::Transpose{T,S}, x::Vector,
                     k::Integer, beta) where {T,S<:DenseMatrix}

This function updates x during the kaczmarz algorithm for dense matrices.
"""
function kaczmarz_update!(B::Transpose{T,S}, x::V,
  k::Integer, beta) where {T,S<:DenseMatrix,V<:DenseVector}
  A = B.parent
  @inbounds @simd for n = 1:size(A, 1)
    x[n] += beta * conj(A[n, k])
  end
end

function kaczmarz_update!(prod::ProdOp{Tc, WeightingOp{T, vecT}}, x, k, beta) where {T, Tc<:Union{T, Complex{T}}, vecT}
  weight = prod.A.weights[k]
  kaczmarz_update!(prod.B, x, k, weight*beta) # only for real weights
end

# kaczmarz_update! with manual simd optimization
for (T, W, WS, shufflevectorMask, vσ) in [(Float32, :WF32, :WF32S, :shufflevectorMaskF32, :vσF32), (Float64, :WF64, :WF64S, :shufflevectorMaskF64, :vσF64)]
  eval(quote
    const $WS = VectorizationBase.pick_vector_width($T)
    const $W = Int(VectorizationBase.pick_vector_width($T))
    const $shufflevectorMask = Val(ntuple(k -> iseven(k - 1) ? k : k - 2, $W))
    const $vσ = Vec(ntuple(k -> (-1.0f0)^(k + 1), $W)...)
    function kaczmarz_update!(A::Transpose{Complex{$T},S}, b::V, k::Integer, beta::Complex{$T}) where {S<:DenseMatrix{Complex{$T}}, V<:DenseVector{Complex{$T}}}
      b = reinterpret($T, b)
      A = reinterpret($T, A.parent)

      N = length(b)
      Nrep, Nrem = divrem(N, 4 * $W) # main loop
      Mrep, Mrem = divrem(Nrem, $W) # last iterations
      idx = MM{$W}(1)
      iOffset = 4 * $W

      vβr = vbroadcast($WS, beta.re) * $vσ # vector containing (βᵣ,-βᵣ,βᵣ,-βᵣ,...)
      vβi = vbroadcast($WS, beta.im) # vector containing (βᵢ,βᵢ,βᵢ,βᵢ,...)

      GC.@preserve b A begin # protect A and y from GC
        vptrA = stridedpointer(A)
        vptrb = stridedpointer(b)
        for _ = 1:Nrep
          Base.Cartesian.@nexprs 4 i -> vb_i = vload(vptrb, ($W * (i - 1) + idx,))
          Base.Cartesian.@nexprs 4 i -> va_i = vload(vptrA, ($W * (i - 1) + idx, k))
          Base.Cartesian.@nexprs 4 i -> begin
            vb_i = muladd(va_i, vβr, vb_i)
            va_i = shufflevector(va_i, $shufflevectorMask)
            vb_i = muladd(va_i, vβi, vb_i)
            vstore!(vptrb, vb_i, ($W * (i - 1) + idx,))
          end
          idx += iOffset
        end

        for _ = 1:Mrep
          vb = vload(vptrb, (idx,))
          va = vload(vptrA, (idx, k))
          vb = muladd(va, vβr, vb)
          va = shufflevector(va, $shufflevectorMask)
          vb = muladd(va, vβi, vb)
          vstore!(vptrb, vb, (idx,))
          idx += $W
        end

        if Mrem != 0
          vloadMask = VectorizationBase.mask($T, Mrem)
          vb = vload(vptrb, (idx,), vloadMask)
          va = vload(vptrA, (idx, k), vloadMask)
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
  N = A.colptr[k+1] - A.colptr[k]
  for n = A.colptr[k]:N-1+A.colptr[k]
    @inbounds x[A.rowval[n]] += beta * conj(A.nzval[n])
  end
end
