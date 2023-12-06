export cgnr, CGNR

mutable struct CGNR{matT,opT,vecT,T,R,PR} <: AbstractKrylovSolver
  A::matT
  AHA::opT
  L2::R
  constr::PR
  x::vecT
  x₀::vecT
  pl::vecT
  vl::vecT
  αl::T
  βl::T
  ζl::T
  weights::vecT
  iterations::Int64
  relTol::Float64
  z0::Float64
  normalizeReg::AbstractRegularizationNormalization
end

"""
    CGNR(A; AHA = A' * A, reg = L2Regularization(zero(real(eltype(AHA)))), normalizeReg = NoNormalization(), weights = similar(AHA, 0), iterations = 10, relTol = eps(real(eltype(AHA))))
    CGNR( ; AHA = ,       reg = L2Regularization(zero(real(eltype(AHA)))), normalizeReg = NoNormalization(), weights = similar(AHA, 0), iterations = 10, relTol = eps(real(eltype(AHA))))

creates an `CGNR` object for the forward operator `A` or normal operator `AHA`.

# Required Arguments
  * `A`                                                 - forward operator
  OR
  * `AHA`                                               - normal operator (as a keyword argument)

# Optional Keyword Arguments
  * `AHA`                                               - normal operator is optional if `A` is supplied
  * `reg::AbstractParameterizedRegularization`          - regularization term; can also be a vector of regularization terms
  * `normalizeReg::AbstractRegularizationNormalization` - regularization normalization scheme; options are `NoNormalization()`, `MeasurementBasedNormalization()`, `SystemMatrixBasedNormalization()`
  * `weights::AbstactVector`                            - weights for the data term; must be of same length and type as the data term
  * `iterations::Int`                                   - maximum number of iterations
  * `relTol::Real`                                      - tolerance for stopping criterion

See also [`createLinearSolver`](@ref), [`solve!`](@ref).
"""
CGNR(; AHA = A'*A, reg = L2Regularization(zero(real(eltype(AHA)))), normalizeReg::AbstractRegularizationNormalization = NoNormalization(), weights::AbstractVector = similar(AHA, 0), iterations::Int = 10, relTol::Real = eps(real(eltype(AHA)))) = CGNR(nothing; AHA, reg, normalizeReg, weights, iterations, relTol)

function CGNR(A
            ; AHA = A'*A
            , reg = L2Regularization(zero(real(eltype(AHA))))
            , normalizeReg::AbstractRegularizationNormalization = NoNormalization()
            , weights::AbstractVector = similar(AHA, 0)
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
  reg = vec(reg)
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


  return CGNR(A, AHA,
    L2, other, x, x₀, pl, vl, αl, βl, ζl, weights, iterations, relTol, 0.0, normalizeReg)
end

"""
    init!(solver::CGNR, b; x0 = 0)

(re-) initializes the CGNR iterator
"""
function init!(solver::CGNR, b; x0 = 0)
  solver.pl .= 0     #temporary vector
  solver.vl .= 0     #temporary vector
  solver.αl  = 0     #temporary scalar
  solver.βl  = 0     #temporary scalar
  solver.ζl  = 0     #temporary scalar
  if all(x0 .== 0)
    solver.x .= 0
  else
    solver.A === nothing && error("providing x0 requires solver.A to be defined")
    solver.x .= x0
    mul!(b, solver.A, solver.x, -1, 1)
  end

  #x₀ = Aᶜ*rl, where ᶜ denotes complex conjugation
  if solver.A === nothing
    !isempty(solver.weights) && @info "weights are being ignored if the backprojection is pre-computed"
    solver.x₀ .= b
  else
    if isempty(solver.weights)
      mul!(solver.x₀, adjoint(solver.A), b)
    else
      mul!(solver.x₀, adjoint(solver.A), b .* solver.weights)
    end
  end

  solver.z0 = norm(solver.x₀)
  copyto!(solver.pl, solver.x₀)

  # normalization of regularization parameters
  solver.L2 = normalize(solver, solver.normalizeReg, solver.L2, solver.A, b)
end


"""
  iterate(solver::CGNR{vecT,T,Tsparse}, iteration::Int=0) where {vecT,T,Tsparse}

performs one CGNR iteration.
"""
function iterate(solver::CGNR, iteration::Int=0)
  if done(solver, iteration)
    for r in solver.constr
      prox!(r, solver.x)
    end
    return nothing
  end

  mul!(solver.vl, solver.AHA, solver.pl)

  solver.ζl = norm(solver.x₀)^2
  normvl = dot(solver.pl, solver.vl)

  λ_ = λ(solver.L2)
  if λ_ > 0
    solver.αl = solver.ζl / (normvl + λ_ * norm(solver.pl)^2)
  else
    solver.αl = solver.ζl / normvl
  end

  BLAS.axpy!(solver.αl, solver.pl, solver.x)

  BLAS.axpy!(-solver.αl, solver.vl, solver.x₀)

  if λ_ > 0
    BLAS.axpy!(-λ_ * solver.αl, solver.pl, solver.x₀)
  end

  solver.βl = dot(solver.x₀, solver.x₀) / solver.ζl

  rmul!(solver.pl, solver.βl)
  BLAS.axpy!(one(eltype(solver.AHA)), solver.x₀, solver.pl)
  return solver.x₀, iteration + 1
end


function converged(solver::CGNR)
  return norm(solver.x₀) / solver.z0 <= solver.relTol
end

@inline done(solver::CGNR, iteration::Int) = converged(solver) || iteration >= min(solver.iterations, size(solver.AHA, 2))