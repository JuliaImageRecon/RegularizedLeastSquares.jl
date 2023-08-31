export Regularization, AbstractRegularization, AbstractParameterizedRegularization, AbstractProjectionRegularization, lambdList, prox!, sink, λ

abstract type AbstractRegularization end
abstract type AbstractParameterizedRegularization{T} <: AbstractRegularization end
sink(reg::AbstractRegularization) = reg
prox!(reg::AbstractParameterizedRegularization, x::AbstractArray) = prox!(reg, x, λ(reg))
norm(reg::AbstractParameterizedRegularization, x::AbstractArray) = norm(reg, x, λ(reg))
λ(reg::AbstractParameterizedRegularization) = reg.λ
# Conversion
prox!(reg::AbstractParameterizedRegularization, x::AbstractArray{Tc}, λ) where {T, Tc<:Union{T, Complex{T}}} = prox!(reg, x, convert(T, λ))
norm(reg::AbstractParameterizedRegularization, x::AbstractArray{Tc}, λ) where {T, Tc<:Union{T, Complex{T}}} = norm(reg, x, convert(T, λ))

prox!(str::AbstractString, x, λ; kwargs...) = prox!(Regularization(str, λ; kwargs...), x, λ)
norm(str::AbstractString, x, λ; kwargs...) = norm(Regularization(str, λ; kwargs...), x, λ)
prox!(regType::Type{<:AbstractParameterizedRegularization}, x, λ; kwargs...) = prox!(regType(λ; kwargs...), x, λ)
norm(regType::Type{<:AbstractParameterizedRegularization}, x, λ; kwargs...) = norm(regType(λ; kwargs...), x, λ)



abstract type AbstractProjectionRegularization <: AbstractRegularization end
prox!(::R, x::AbstractArray) where {R<:AbstractProjectionRegularization} = prox!(R, x)
norm(::R, x::AbstractArray) where {R<:AbstractProjectionRegularization} = norm(R, x)
λ(::AbstractProjectionRegularization) = nothing

prox!(regType::Type{<:AbstractProjectionRegularization}, x; kwargs...) = prox!(regType(;kwargs...), x)
norm(regType::Type{<:AbstractProjectionRegularization}, x; kwargs...) = norm(regType(;kwargs...), x)

export AbstractRegularizationNormalization, NoNormalization, MeasurementBasedNormalization, SystemMatrixBasedNormalization
abstract type AbstractRegularizationNormalization end
struct NoNormalization <: AbstractRegularizationNormalization end
struct MeasurementBasedNormalization <: AbstractRegularizationNormalization end
struct SystemMatrixBasedNormalization <: AbstractRegularizationNormalization end
# TODO weighted systemmatrix, maybe weighted measurementbased?

struct NormalizedRegularization{TF, R<:AbstractRegularization} <: AbstractParameterizedRegularization{TF}
  reg::R
  factor::TF
end
sink(reg::NormalizedRegularization) = sink(reg.reg)
λ(reg::NormalizedRegularization) = λ(reg.reg) * reg.factor
prox!(reg::NormalizedRegularization, x::AbstractArray, λ) = prox!(reg.reg, x, λ)
norm(reg::NormalizedRegularization, x::AbstractArray, λ) = norm(reg.reg, x, λ)

function normalize(::MeasurementBasedNormalization, A, b::AbstractArray)
  return norm(b, 1)/length(b)
end
normalize(::MeasurementBasedNormalization, A, b::Nothing) = normalize(NoNormalization(), A, b)
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

Base.vec(reg::AbstractRegularization) = AbstractRegularization[reg]
Base.vec(reg::Vector{AbstractRegularization}) = reg

"""
    RegularizationList()

Returns a list of all available Regularizations
"""
function RegularizationList()
  return subtypes(RegularizationList)
end

norm0(x::Array{T}, λ::T; sparseTrafo=nothing, kargs...) where T = 0.0
