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

# Moving a solver to another array backend

    Adapt.adapt(target, solver::CGNR)

adapts the operators, regularization and state of the solver to `target` (e.g. `CuArray`). Adapt the solver to the
target backend before solving; each operator must support Adapt.jl. Of the solvers, `CGNR`, `ADMM`, `DirectSolver` and `PseudoInverse` support `Adapt.adapt` so far.

See also [`createLinearSolver`](@ref), [`solve!`](@ref).
"""
CGNR(; AHA, kwargs...) = CGNR(nothing; AHA = AHA, kwargs...)

function CGNR(A
            ; AHA = A'*A
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

function Adapt.adapt_structure(to, solver::CGNR)
  A = Adapt.adapt(to, solver.A)
  AHA = Adapt.adapt(to, solver.AHA)
  L2 = Adapt.adapt(to, solver.L2)
  # adapt each term, not the vector itself, which would become an array of terms on the target
  constr = map(reg -> Adapt.adapt(to, reg), solver.constr)
  normalizeReg = Adapt.adapt(to, solver.normalizeReg)
  state = Adapt.adapt(to, solver.state)
  return CGNR(A, AHA, L2, constr, normalizeReg, solver.iterations, state)
end

Adapt.adapt_structure(to, state::CGNRState) = CGNRState(Adapt.adapt(to, state.x), Adapt.adapt(to, state.x₀), Adapt.adapt(to, state.pl), Adapt.adapt(to, state.vl),
  state.αl, state.βl, state.ζl, state.iteration, state.relTol, state.z0)
