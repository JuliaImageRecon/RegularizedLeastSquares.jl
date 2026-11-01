export OSSART

mutable struct OSSART{matT, B, W, T, R, RN} <: AbstractRowActionSolver
  A::matT
  blocks::B
  invrowsums::W
  invcolsums::W
  relaxation::T
  constr::Vector{R}
  reg::Vector{RN}
  normalizeReg::AbstractRegularizationNormalization
  iterations::Int64
  state::AbstractSolverState{<:OSSART}
end

mutable struct OSSARTState{T, vecT <: AbstractArray} <: AbstractSolverState{OSSART}
  x::vecT
  b::vecT
  r::vecT
  dx::vecT
  iteration::Int64
  ressq::T
end

"""
    OSSART(A; blocks = RowBlockOp([A]), weights = nothing, relaxation = 1, reg = nothing, normalizeReg = NoNormalization(), iterations = 10)

creates an `OSSART` object for the forward operator `A`: the ordered-subset simultaneous algebraic reconstruction
technique (Wang & Jiang 2004), with SIRT (one block, Gilbert 1972) and SART (blocks of one view, Andersen & Kak 1984)
as special cases. The rows of `A` are split into blocks `A_k`; one iteration is a pass through all blocks, each an
update

    x ← x + ω V_k⁻¹ A_kᴴ W_k⁻¹ (b_k − A_k x)

with `W_k` the row sums and `V_k` the column sums of `A_k` (their inverses zero where a sum is zero) and the
relaxation `ω ∈ (0, 2)`. For consistent data it converges to a solution, otherwise to a weighted least-squares
solution. The weights assume nonnegative entries, as in projectors and system matrices of emission tomography; for
matrices the sums of absolute values are used.

The blocks are a `RowBlockOp` of LinearOperatorCollection.jl with the size of `A`: the operators of the blocks with
their rows in `b`, e.g. `RowBlockOp(A, 10)` for 10 blocks of interleaved rows of a matrix. Without `blocks`, `A` is a
single block (SIRT).

# Required Arguments
  * `A`                                                 - forward operator

# Optional Keyword Arguments
  * `blocks::RowBlockOp`                                - the blocks of rows of `A`
  * `weights`                                           - vector of `(rowsums, colsums)` per block; default: computed from the blocks (for operators that are no matrices as `op * ones` and `adjoint(op) * ones`, which requires nonnegative entries)
  * `relaxation::Real`                                  - relaxation parameter `ω ∈ (0, 2)`
  * `reg::AbstractRegularization`                       - regularization terms: projections (`PositiveRegularization`, `RealRegularization`) after each block, and at most one further term, applied with its proximal map after each pass (as in `Kaczmarz`). The alternation of OS-SART passes and a proximal step is a heuristic, not a minimization of data term plus regularization; this holds for L2 as well, whose proximal map shrinks `x` after each pass but does not give the Tikhonov solution that `CGNR` computes.
  * `normalizeReg::AbstractRegularizationNormalization` - regularization normalization scheme; options are `NoNormalization()`, `MeasurementBasedNormalization()`, `SystemMatrixBasedNormalization()`
  * `iterations::Int`                                   - number of passes through all blocks

References:
  * P. Gilbert, "Iterative methods for the three-dimensional reconstruction of an object from projections", J. Theor. Biol. 36(1), 105–117, 1972.
  * A. H. Andersen, A. C. Kak, "Simultaneous algebraic reconstruction technique (SART): a superior implementation of the ART algorithm", Ultrasonic Imaging 6(1), 81–94, 1984.
  * G. Wang, M. Jiang, "Ordered-subset simultaneous algebraic reconstruction techniques (OS-SART)", J. X-Ray Sci. Technol. 12(3), 169–177, 2004.

See also [`createLinearSolver`](@ref), [`solve!`](@ref).
"""
function OSSART(A; blocks::RowBlockOp = RowBlockOp([A]), weights = nothing, relaxation::Real = 1, reg = nothing,
                normalizeReg::AbstractRegularizationNormalization = NoNormalization(), iterations::Int = 10)
  0 < relaxation < 2 || throw(ArgumentError("the relaxation must lie in (0, 2), got $relaxation"))
  size(blocks) == size(A) || throw(DimensionMismatch("blocks of size $(size(blocks)) for an operator of size $(size(A))"))
  T = real(eltype(A))
  blocks = rowblocks(blocks)
  weights = isnothing(weights) ? [sart_weights(op) for (op, _) in blocks] : weights
  length(weights) == length(blocks) || throw(ArgumentError("$(length(weights)) weights for $(length(blocks)) blocks"))
  invrowsums = [inverse_or_zero(T, w[1]) for w in weights]
  invcolsums = [inverse_or_zero(T, w[2]) for w in weights]

  # Prepare regularization terms
  reg = isnothing(reg) ? AbstractRegularization[] : copy(isa(reg, AbstractVector) ? reg : [reg])
  reg = normalize(OSSART, normalizeReg, reg, A, nothing)
  indices = findsinks(AbstractProjectionRegularization, reg)
  constr = identity.(AbstractRegularization[reg[i] for i in indices])
  deleteat!(reg, indices)
  length(reg) > 1 && error("OSSART does not allow for more than one additional regularization term, found $(length(reg))")
  other = identity.(AbstractRegularization[r for r in reg])

  m = maximum(length(rows) for (_, rows) in blocks)
  x = zeros(eltype(A), size(A, 2))
  state = OSSARTState(x, zeros(eltype(A), size(A, 1)), zeros(eltype(A), m), similar(x), 0, zero(T))
  return OSSART(A, blocks, invrowsums, invcolsums, T(relaxation), constr, other, normalizeReg, iterations, state)
end

# the row and column sums of a block: of the absolute values for matrices, else `op * ones` and `adjoint(op) * ones`
sart_weights(op::AbstractMatrix) = (vec(sum(abs, op; dims = 2)), vec(sum(abs, op; dims = 1)))
function sart_weights(op)
  T = eltype(op)
  S = LinearOperatorCollection.LinearOperators.storage_type(op)
  rowsums = op * fill!(S(undef, size(op, 2)), one(T))
  colsums = adjoint(op) * fill!(S(undef, size(op, 1)), one(T))
  return (abs.(rowsums), abs.(colsums))
end

inverse_or_zero(::Type{T}, w) where T = map(v -> v > 0 ? inv(T(real(v))) : zero(T), w)

function init!(solver::OSSART, state::OSSARTState{T, vecT}, b::otherT; kwargs...) where {T, vecT, otherT <: AbstractVector}
  x = similar(b, size(state.x)...)
  bb = similar(b, size(state.b)...)
  r = similar(b, size(state.r)...)
  dx = similar(b, size(state.dx)...)
  state = OSSARTState(x, bb, r, dx, state.iteration, state.ressq)
  solver.state = state
  init!(solver, state, b; kwargs...)
end

"""
    init!(solver::OSSART, b; x0 = 0)

(re-) initializes the OSSART iterator; `x0` is the start vector.
"""
function init!(solver::OSSART, state::OSSARTState{T, vecT}, b::vecT; x0 = 0) where {T, vecT <: AbstractVector}
  solver.reg = normalize(solver, solver.normalizeReg, solver.reg, solver.A, b)
  state.x .= x0
  state.b .= b
  state.iteration = 0
  state.ressq = zero(T)
end

solverconvergence(state::OSSARTState) = (; :residual => sqrt(state.ressq))

function iterate(solver::OSSART, state::OSSARTState)
  if done(solver, state) return nothing end
  state.ressq = zero(state.ressq)
  for (k, (op, rows)) in enumerate(solver.blocks)
    r = view(state.r, 1:length(rows))
    mul!(r, op, state.x)
    r .= view(state.b, rows) .- r
    state.ressq += real(sum(abs2, r))
    r .*= solver.invrowsums[k]
    mul!(state.dx, adjoint(op), r)
    state.x .+= solver.relaxation .* solver.invcolsums[k] .* state.dx
    for c in solver.constr
      prox!(c, state.x)
    end
  end
  for reg in solver.reg
    prox!(reg, state.x)
  end
  state.iteration += 1
  return state.x, state
end

@inline done(solver::OSSART, state::OSSARTState) = state.iteration >= solver.iterations

function isapplicable(::Type{T}, reg::Vector{<:AbstractRegularization}) where T <: OSSART
  applicable = true
  applicable &= length(findsinks(AbstractParameterizedRegularization, reg)) <= 1
  return applicable
end
