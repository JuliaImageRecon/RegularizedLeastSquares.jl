export AbstractRegularizationNormalization, NormalizedRegularization, NoNormalization, MeasurementBasedNormalization, SystemMatrixBasedNormalization
abstract type AbstractRegularizationNormalization end
"""
    NoNormalization
  
No normalization to `λ` is applied.
"""
struct NoNormalization <: AbstractRegularizationNormalization end
"""
    MeasurementBasedNormalization

`λ` is normalized by the 1-norm of `b` divided by its length.
"""
struct MeasurementBasedNormalization <: AbstractRegularizationNormalization end
"""
    SystemMatrixBasedNormalization

`λ` is normalized by the energy of the system matrix rows.
"""
struct SystemMatrixBasedNormalization <: AbstractRegularizationNormalization end
# TODO weighted systemmatrix, maybe weighted measurementbased?

"""
    NormalizedRegularization

Nested regularization term that scales `λ` according to normalization scheme. This term is commonly applied by a solver based on a given normalization keyword

#See also [`NoNormalization`](@ref), [`MeasurementBasedNormalization`](@ref), [`SystemMatrixBasedNormalization`](@ref).
"""
struct NormalizedRegularization{T, S, R} <: AbstractScaledRegularization{T, S}
  reg::R
  factor::T
  NormalizedRegularization(reg::R, factor) where {T, R <: AbstractParameterizedRegularization{T}} = new{T, R, R}(reg, factor)
  NormalizedRegularization(reg::R, factor) where {T, RN <: AbstractParameterizedRegularization{T}, R<:AbstractNestedRegularization{RN}} = new{T, RN, R}(reg, factor)
end
innerreg(reg::NormalizedRegularization) = reg.reg
scalefactor(reg::NormalizedRegularization) = reg.factor

function normalize(::MeasurementBasedNormalization, A, b::AbstractArray)
  return norm(b, 1)/length(b)
end
normalize(::MeasurementBasedNormalization, A, b::Nothing) = one(real(eltype(A)))
function normalize(::SystemMatrixBasedNormalization, A::AbstractArray{T}, b) where {T}
  M = size(A, 1)
  N = size(A, 2)

  energy = zeros(T, M)
  for m=1:M
    energy[m] = sqrt(rownorm²(A,m))
  end

  trace = norm(energy)^2/N
  # TODO where setlamda? here we dont know λ
  return trace
end
normalize(::NoNormalization, A, b) = nothing
function normalize(norm::AbstractRegularizationNormalization, regs::Vector{R}, A, b) where {R<:AbstractRegularization}
  factor = normalize(norm, A, b)
  return map(x-> normalize(x, factor), regs)
end
function normalize(norm::AbstractRegularizationNormalization, reg::R, A, b) where {R<:AbstractRegularization}
  factor = normalize(norm, A, b)
  return normalize(reg, factor)
end

normalize(reg::R, ::Nothing) where {R<:AbstractRegularization} = reg
normalize(reg::AbstractProjectionRegularization, factor::Number) = reg
normalize(reg::NormalizedRegularization, factor::Number) = NormalizedRegularization(reg.reg, factor) # Update normalization
normalize(reg::AbstractParameterizedRegularization, factor::Number) = NormalizedRegularization(reg, factor)
function normalize(reg::AbstractRegularization, factor::Number)
  if sink(reg) isa AbstractParameterizedRegularization
    return NormalizedRegularization(reg, factor)
  end
  return reg
end


normalize(solver::AbstractLinearSolver, norm, regs, A, b) = normalize(typeof(solver), norm, regs, A, b)
normalize(solver::Type{T}, norm::AbstractRegularizationNormalization, regs, A, b) where T<:AbstractLinearSolver = normalize(norm, regs, A, b)
# System matrix based normalization is already done in constructor, can just return regs
normalize(solver::AbstractLinearSolver, norm::SystemMatrixBasedNormalization, regs, A, b) = regs
