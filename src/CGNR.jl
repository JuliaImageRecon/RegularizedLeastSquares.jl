export cgnr, CGNR

mutable struct CGNR{matT,opT, R,PR} <: AbstractKrylovSolver
  A::matT
  AHA::opT
  L2::R
  constr::PR
  normalizeReg::AbstractRegularizationNormalization
  iterations::Int64
  state::AbstractSolverState{<:CGNR}
end

mutable struct CGNRState{T, Tc, vecTc} <: AbstractSolverState{CGNR} where {T, Tc <: Union{T, Complex{T}}, vecTc<:AbstractArray{Tc}}
  x::vecTc
  x₀::vecTc
  pl::vecTc
  vl::vecTc
  αl::Tc
  βl::Tc
  ζl::Tc
  iteration::Int64
  relTol::T
  z0::T
end

"""
    CGNR(A; AHA = A' * A, reg = L2Regularization(zero(real(eltype(AHA)))), normalizeReg = NoNormalization(), iterations = 10, relTol = eps(real(eltype(AHA))))
    CGNR( ; AHA = ,       reg = L2Regularization(zero(real(eltype(AHA)))), normalizeReg = NoNormalization(), iterations = 10, relTol = eps(real(eltype(AHA))))

creates an `CGNR` object for the forward operator `A` or normal operator `AHA`.

# Required Arguments
  * `A`                                                 - forward operator
  OR
  * `AHA`                                               - normal operator (as a keyword argument)

# Optional Keyword Arguments
  * `AHA`                                               - normal operator is optional if `A` is supplied
  * `reg::AbstractParameterizedRegularization`          - regularization term; can also be a vector of regularization terms
  * `normalizeReg::AbstractRegularizationNormalization` - regularization normalization scheme; options are `NoNormalization()`, `MeasurementBasedNormalization()`, `SystemMatrixBasedNormalization()`
  * `iterations::Int`                                   - maximum number of iterations
  * `relTol::Real`                                      - tolerance for stopping criterion

If `A` is an operator from LinearOperatorCollection.jl, the default `AHA` is the structured
`NormalOp(eltype(storage_type(A)); parent = A)`, whose action is exactly `adjoint(A) * (A * x)`.
For other `A` (e.g. dense or sparse matrices) the default is `A' * A`. Specialized normal
operators such as `normalOperator(A)` can be passed explicitly via `AHA`.

# Moving a solver to another array backend

    Adapt.adapt(target, solver::CGNR)

returns a new `CGNR` whose operators, regularization and state live on `target` (e.g. `CuArray`,
`ROCArray` or `JLArray`; `Array` gives an independent CPU copy). Initialized and partially iterated
solvers keep their progress, including frame states created by matrix-valued `init!`/`solve!`
(`SequentialState`, `MultiThreadingState`) and their active flags. Arrays the target would leave in place are
copied, so the result shares no data or workspace with the source solver; only read-only host index
vectors of `DiagOp`/`DiagNormalOp` may be shared. The source solver is not modified.

Supported operators are dense arrays and LinearOperatorCollection.jl operators with Adapt rules. An
`AHA` of type `NormalOp` whose parent is `A` (the default), or the inner operand `M` of a weighted
`ProdOp(W, M)` (as built by `normalOperator(A)`), is rebuilt on the adapted `A`/`M`.
Supported regularization: `L2Regularization`, `RealRegularization`, `PositiveRegularization` and their
normalized, scaled, fixed-parameter, auto-scaled, masked and transformed wrappers.

An `ArgumentError` is thrown instead of returning a partially moved solver for unsupported targets
(e.g. `Int`, `AbstractArray` or an adaptor that leaves the state unchanged), element-type-changing
targets, operators without an Adapt rule (including closure-based `LinearOperator`s and products
such as `A' * A` of operators), and unsupported custom states or regularization terms.

Only `CGNR` supports `Adapt.adapt` so far.

See also [`createLinearSolver`](@ref), [`solve!`](@ref).
"""
CGNR(; AHA, kwargs...) = CGNR(nothing; AHA = AHA, kwargs...)

function CGNR(A
            ; AHA = cgnrDefaultNormal(A)
            , reg = L2Regularization(zero(real(eltype(AHA))))
            , normalizeReg::AbstractRegularizationNormalization = NoNormalization()
            , iterations::Int = 10
            , relTol::Real = eps(real(eltype(AHA)))
            )

  T = eltype(AHA)

  x = Vector{T}(undef, size(AHA, 2))
  x₀ = similar(x)     #temporary vector
  pl = similar(x)     #temporary vector
  vl = similar(x)     #temporary vector
  αl = zero(T)        #temporary scalar
  βl = zero(T)        #temporary scalar
  ζl = zero(T)        #temporary scalar

  # Prepare regularization terms
  reg = copy(isa(reg, AbstractVector) ? reg : [reg])
  reg = normalize(CGNR, normalizeReg, reg, A, nothing)
  idx = findsink(L2Regularization, reg)
  if isnothing(idx)
    L2 = L2Regularization(zero(T))
  else
    L2 = reg[idx]
    deleteat!(reg, idx)
  end

  indices = findsinks(RealRegularization, reg)
  push!(indices, findsinks(PositiveRegularization, reg)...)
  other = [reg[i] for i in indices]
  deleteat!(reg, indices)
  if length(reg) > 0
    error("CGNR does not allow for more additional regularization terms, found $(length(reg))")
  end
  other = identity.(other)

  state = CGNRState(x, x₀, pl, vl, αl, βl, ζl, 0, real(T)(relTol), zero(real(T)))

  return CGNR(A, AHA, L2, other, normalizeReg, iterations, state)
end

function init!(solver::CGNR, state::CGNRState{T, Tc, vecTc}, b::otherTc; kwargs...) where {T, Tc, vecTc, otherTc <: AbstractVector{Tc}}
  x = similar(b, size(state.x)...)
  x₀ = similar(b, size(state.x₀)...)
  pl = similar(b, size(state.pl)...)
  vl = similar(b, size(state.vl)...)

  state = CGNRState(x, x₀, pl, vl, state.αl, state.βl, state.ζl, state.iteration, state.relTol, state.z0)
  solver.state = state
  init!(solver, state, b; kwargs...)
end

"""
    init!(solver::CGNR, b; x0 = 0)

(re-) initializes the CGNR iterator
"""
function init!(solver::CGNR, state::CGNRState{T, Tc, vecTc}, b::vecTc; x0 = 0) where {T, Tc <: Union{T, Complex{T}}, vecTc<:AbstractVector{Tc}}
  state.pl .= 0     #temporary vector
  state.vl .= 0     #temporary vector
  state.αl  = 0     #temporary scalar
  state.βl  = 0     #temporary scalar
  state.ζl  = 0     #temporary scalar
  state.iteration = 0

  # normalization of regularization parameters (before λ is used for the start residual)
  solver.L2 = normalize(solver, solver.normalizeReg, solver.L2, solver.A, b)

  #x₀ = Aᶜ*rl, where ᶜ denotes complex conjugation
  initCGNR(state.x₀, solver.A, b)

  # with a start value x0 the residual of the normal equation is Aᶜ*b - (AᶜA + λI)*x0; computed
  # with the normal operator, so it works without A and leaves b unchanged
  if all(x0 .== 0)
    state.x .= 0
  else
    state.x .= x0
    mul!(state.vl, solver.AHA, state.x)
    state.x₀ .-= state.vl
    λ_ = λ(solver.L2)
    if λ_ > 0
      state.x₀ .-= λ_ .* state.x
    end
    state.vl .= 0
  end

  state.z0 = norm(state.x₀)
  copyto!(state.pl, state.x₀)
end

initCGNR(x₀, A, b) = mul!(x₀, adjoint(A), b)
#initCGNR(x₀, prod::ProdOp{T, <:WeightingOp, matT}, b) where {T, matT} = mul!(x₀, adjoint(prod.B), b)
initCGNR(x₀, ::Nothing, b) = x₀ .= b

solverconvergence(state::CGNRState) = (; :residual => norm(state.x₀))

"""
  iterate(solver::CGNR{vecT,T,Tsparse}, iteration::Int=0) where {vecT,T,Tsparse}

performs one CGNR iteration.
"""
function iterate(solver::CGNR, state::CGNRState)
  if done(solver, state)
    for r in solver.constr
      prox!(r, state.x)
    end
    return nothing
  end

  mul!(state.vl, solver.AHA, state.pl)

  state.ζl = norm(state.x₀)^2
  normvl = dot(state.pl, state.vl)

  λ_ = λ(solver.L2)
  if λ_ > 0
    state.αl = state.ζl / (normvl + λ_ * norm(state.pl)^2)
  else
    state.αl = state.ζl / normvl
  end

  state.x .+= state.pl .* state.αl

  state.x₀ .+= state.vl .* -state.αl

  if λ_ > 0
    state.x₀ .+= state.pl .* -λ_ * state.αl
  end

  state.βl = dot(state.x₀, state.x₀) / state.ζl

  rmul!(state.pl, state.βl)
  state.pl .+= state.x₀

  state.iteration += 1
  return state.x, state
end


function converged(::CGNR, state::CGNRState)
  return norm(state.x₀) / state.z0 <= state.relTol
end

@inline done(solver::CGNR, state::CGNRState) = converged(solver, state) || state.iteration >= min(solver.iterations, size(solver.AHA, 2))

# Default normal operator: structured (and therefore adaptable) for LinearOperatorCollection operators.
cgnrDefaultNormal(A) = A' * A
cgnrDefaultNormal(A::LinearOperatorCollection.AbstractLinearOperatorFromCollection) = NormalOp(eltype(LinearOperators.storage_type(A)); parent = A)

########################################
# Adapt: rebuild a CGNR solver on another array backend
########################################

struct CGNRAdaptContext{TO, F}
  to::TO
  family::F                     # array type constructor of the target backend, e.g. JLArray
  visited::IdDict{Any, Any}     # source node => adapted node, for every checked node
  memo::IdDict{Any, Any}        # source component => adapted component (keeps A === AHA etc.)
end

adaptError(msg) = throw(ArgumentError("cannot adapt CGNR: " * msg))

baseArray(x::AbstractArray) = (p = parent(x); p === x ? x : baseArray(p))
arrayFamily(x::AbstractArray) = Base.typename(typeof(baseArray(x))).wrapper

# Wraps the user's adaptor and copies every numeric array it leaves in place (e.g. `Array` on a CPU
# solver), so that operators rebuilt by their Adapt rules get their own workspaces instead of sharing
# them, and the closures capturing them, with the source solver.
struct CGNRFreshArrays{TO}
  to::TO
end
function Adapt.adapt_storage(a::CGNRFreshArrays, x)
  y = Adapt.adapt(a.to, x)
  return y === x && x isa AbstractArray{<:Number} ? copy(x) : y
end
typeFamily(S::Type) = Base.typename(Base.unwrap_unionall(S)).wrapper

cgnrStates(state::CGNRState) = [state]
cgnrStates(state::Union{SequentialState, MultiThreadingState}) = state.states
cgnrStates(state) = adaptError("unsupported solver state $(typeof(state))")

function Adapt.adapt_structure(to, solver::CGNR)
  states = cgnrStates(solver.state)
  isempty(states) && adaptError("the solver state contains no CGNR states")
  first(states) isa CGNRState || adaptError("unsupported solver state $(typeof(first(states)))")
  x = first(states).x
  to isa Type && isabstracttype(to) && adaptError("unsupported Adapt target $to (abstract type)")
  x′ = Adapt.adapt(to, x)
  supported = x′ isa AbstractVector && eltype(x′) == eltype(x) && (to isa Type ? x′ isa to : x′ !== x)
  supported || adaptError("unsupported Adapt target $to: Adapt.adapt returned $(typeof(x′)) for a state vector of type $(typeof(x))")
  ctx = CGNRAdaptContext(CGNRFreshArrays(to), arrayFamily(x′), IdDict{Any, Any}(), IdDict{Any, Any}())

  A = adaptComponent(ctx, solver.A, "A")
  AHA = adaptNormal(ctx, solver.A, A, solver.AHA)
  L2 = adaptReg(ctx, solver.L2)
  constr = isempty(solver.constr) ? copy(solver.constr) : identity.([adaptReg(ctx, reg) for reg in solver.constr])
  state = adaptState(ctx, solver.state)
  return CGNR(A, AHA, L2, constr, solver.normalizeReg, solver.iterations, state)
end

function adaptComponent(ctx::CGNRAdaptContext, src, name)
  isnothing(src) && return nothing
  haskey(ctx.memo, src) && return ctx.memo[src]
  dst = try
    Adapt.adapt(ctx.to, src)
  catch e
    e isa LinearOperators.LinearOperatorException || rethrow()
    adaptError("$name of type $(typeof(src)) could not be adapted: $(sprint(showerror, e))")
  end
  checkPair(ctx, src, dst, name)
  return ctx.memo[src] = dst
end

# A NormalOp that refers to A (or to the inner operand of a weighted ProdOp A) is rebuilt on the adapted
# operand, so that the adapted AHA keeps acting on the adapted forward operator. Its weights are its own.
function adaptNormal(ctx::CGNRAdaptContext, A, A′, AHA)
  if AHA isa LinearOperatorCollection.NormalOp && !haskey(ctx.memo, AHA) && !isnothing(A)
    parent′ = if AHA.parent === A
      A′
    elseif A isa LinearOperatorCollection.ProdOp && AHA.parent === A.B
      A′.B
    else
      nothing
    end
    if !isnothing(parent′)
      weights′ = adaptComponent(ctx, AHA.weights, "AHA weights")
      AHA′ = NormalOp(eltype(AHA); parent = parent′, weights = weights′)
      checkPair(ctx, AHA, AHA′, "AHA")
      return ctx.memo[AHA] = AHA′
    end
  end
  return adaptComponent(ctx, AHA, "AHA")
end

function adaptVector(ctx::CGNRAdaptContext, v, name)
  v′ = Adapt.adapt(ctx.to, v)
  checkArray(ctx, v, v′, name)
  return v′
end

adaptState(ctx::CGNRAdaptContext, state::CGNRState) =
  CGNRState(adaptVector(ctx, state.x, "state.x"), adaptVector(ctx, state.x₀, "state.x₀"),
            adaptVector(ctx, state.pl, "state.pl"), adaptVector(ctx, state.vl, "state.vl"),
            state.αl, state.βl, state.ζl, state.iteration, state.relTol, state.z0)
function adaptState(ctx::CGNRAdaptContext, state::S) where S <: Union{SequentialState, MultiThreadingState}
  adapted = Base.typename(S).wrapper([adaptState(ctx, child) for child in state.states])
  adapted.active .= state.active
  return adapted
end
adaptState(ctx::CGNRAdaptContext, state) = adaptError("unsupported solver state $(typeof(state))")

adaptReg(ctx::CGNRAdaptContext, reg::L2Regularization{<:Number}) = L2Regularization(reg.λ)
adaptReg(ctx::CGNRAdaptContext, ::RealRegularization) = RealRegularization()
adaptReg(ctx::CGNRAdaptContext, ::PositiveRegularization) = PositiveRegularization()
adaptReg(ctx::CGNRAdaptContext, reg::NormalizedRegularization) = NormalizedRegularization(adaptReg(ctx, reg.reg), reg.factor)
adaptReg(ctx::CGNRAdaptContext, reg::FixedScaledRegularization) = FixedScaledRegularization(adaptReg(ctx, reg.reg), reg.factor)
adaptReg(ctx::CGNRAdaptContext, reg::FixedParameterRegularization) = FixedParameterRegularization(adaptReg(ctx, reg.reg))
function adaptReg(ctx::CGNRAdaptContext, reg::AutoScaledRegularization)
  adapted = AutoScaledRegularization(adaptReg(ctx, reg.reg))
  adapted.factor = reg.factor
  return adapted
end
adaptReg(ctx::CGNRAdaptContext, reg::MaskedRegularization) = MaskedRegularization(adaptReg(ctx, reg.reg), copy(reg.mask))
adaptReg(ctx::CGNRAdaptContext, reg::TransformedRegularization) =
  TransformedRegularization(adaptReg(ctx, reg.reg), adaptComponent(ctx, reg.trafo, "regularization transform"))
adaptReg(ctx::CGNRAdaptContext, reg) = adaptError("unsupported regularization term $(typeof(reg))")

# Validation of an adapted operator graph against its source, node by node.
# Records src => dst; returns false if the pair was already checked. Operators shared in the source must
# stay shared; an array shared by several source operators may be adapted into separate copies.
function visit!(ctx::CGNRAdaptContext, src, dst, path)
  if haskey(ctx.visited, src)
    src isa AbstractLinearOperator && ctx.visited[src] !== dst &&
      adaptError("$path: operator of type $(typeof(src)) is shared in the source graph but was adapted into separate objects")
    return false
  end
  ctx.visited[src] = dst
  return true
end

function checkArray(ctx::CGNRAdaptContext, src::AbstractArray, dst, path; host = false)
  (dst isa AbstractArray && eltype(dst) == eltype(src)) ||
    adaptError("$path: array of type $(typeof(src)) was adapted to $(typeof(dst)) (element type must be preserved)")
  visit!(ctx, src, dst, path)
  host && return nothing
  baseArray(dst) !== baseArray(src) ||
    adaptError("$path: array of type $(typeof(src)) would share memory with the source solver")
  arrayFamily(dst) === ctx.family ||
    adaptError("$path: array of type $(typeof(dst)) is not on the target backend $(ctx.family)")
  return nothing
end

isHostIndexField(op, field) = (op isa LinearOperatorCollection.DiagOp && field in (:xIdx, :yIdx)) || (op isa LinearOperatorCollection.DiagNormalOp && field === :idx)

function checkPair(ctx::CGNRAdaptContext, src, dst, path)
  if src isa AbstractLinearOperator
    (dst isa AbstractLinearOperator && Base.typename(typeof(dst)) === Base.typename(typeof(src))) ||
      adaptError("$path: cannot verify adapted operator, $(typeof(src)) became $(typeof(dst))")
    dst !== src || adaptError("$path: operator of type $(typeof(src)) has no Adapt rule (opaque operators, " *
                              "e.g. closure-based LinearOperators or products such as A'*A, cannot be rebuilt)")
    visit!(ctx, src, dst, path) || return nothing
    S, S′ = LinearOperators.storage_type(src), LinearOperators.storage_type(dst)
    (eltype(dst) == eltype(src) && eltype(S′) == eltype(S)) ||
      adaptError("$path: operator of type $(typeof(src)) changed element type to $(typeof(dst))")
    typeFamily(S′) === ctx.family ||
      adaptError("$path: operator of type $(typeof(src)) has storage $S′, not on the target backend $(ctx.family)")
    for field in fieldnames(typeof(src))
      (isdefined(src, field) && isdefined(dst, field)) || continue
      a, b = getfield(src, field), getfield(dst, field)
      if field in (:plan_args, :plan_kwargs) && a isa AbstractDict
        checkReplay(ctx, a, b, "$path.$field")
      elseif a isa AbstractArray{<:Number}
        checkArray(ctx, a, b, "$path.$field"; host = isHostIndexField(src, field))
      else
        checkPair(ctx, a, b, "$path.$field")
      end
    end
  elseif src isa AbstractArray{<:Number}
    checkArray(ctx, src, dst, path)
  elseif src isa Union{AbstractArray, Tuple}
    # host containers of child operators, e.g. DiagOp.ops
    (dst isa Union{AbstractArray, Tuple} && length(dst) == length(src)) ||
      adaptError("$path: cannot verify adapted container, $(typeof(src)) became $(typeof(dst))")
    for (i, (a, b)) in enumerate(zip(src, dst))
      checkPair(ctx, a, b, "$path[$i]")
    end
  end
  return nothing
end

# Constructor inputs that LinearOperatorCollection operators retain to rebuild themselves (e.g. NFFT nodes)
function checkReplay(ctx::CGNRAdaptContext, src::AbstractDict, dst, path)
  (dst isa AbstractDict && Set(keys(dst)) == Set(keys(src))) ||
    adaptError("$path: cannot verify adapted reconstruction inputs")
  for (key, a) in src
    b = dst[key]
    if a isa AbstractDict
      checkReplay(ctx, a, b, "$path[$(repr(key))]")
    elseif a isa AbstractArray{<:Number}
      (b isa AbstractArray && eltype(b) == eltype(a)) ||
        adaptError("$path[$(repr(key))]: $(typeof(a)) was adapted to $(typeof(b)) (element type must be preserved)")
    end
  end
  return nothing
end
