export Regularization, AbstractRegularization, lambdList, prox! #, norm

abstract type AbstractRegularization end
abstract type AbstractParameterizedRegularization{T} <: AbstractRegularization end
prox!(reg::AbstractParameterizedRegularization, x::AbstractArray) = prox!(reg, x, λ(reg))
norm(reg::AbstractParameterizedRegularization, x::AbstractArray) = norm(reg, x, λ(reg))
λ(reg::AbstractParameterizedRegularization) = reg.λ
# Conversion
prox!(reg, x::AbstractArray{Complex{T}}, λ) where {T} = prox!(reg, x, convert(T, λ))
norm(reg, x::AbstractArray{Complex{T}}, λ) where {T} = norm(reg, x, convert(T, λ))
prox!(reg, x::AbstractArray{T}, λ) where {T} = prox!(reg, x, convert(T, λ))
norm(reg, x::AbstractArray{T}, λ) where {T} = norm(reg, x, convert(T, λ))

@generated function prox!(reg::R, x::AbstractArray{Tc}, λ::T) where {R<:AbstractParameterizedRegularization, T, Tc<: Union{T, Complex{T}}}
  kwargs = [Expr(:kw, :($field), :(reg.$field)) for field in filter(x-> x != :λ, fieldnames(R))]
  return Expr(:call, :prox!, Expr(:parameters, kwargs...), reg, :x, :λ)
end

@generated function norm(reg::R, x::AbstractArray{Tc}, λ::T) where {R<:AbstractParameterizedRegularization, T, Tc<: Union{T, Complex{T}}}
  kwargs = [Expr(:kw, :($field), :(reg.$field)) for field in filter(x-> x != :λ, fieldnames(R))]
  return Expr(:call, :norm, Expr(:parameters, kwargs...), reg, :x, :λ)
end

abstract type AbstractProjectionRegularization <: AbstractRegularization end
prox!(::R, x::AbstractArray) where {R<:AbstractProjectionRegularization} = prox!(R, x)
norm(::R, x::AbstractArray) where {R<:AbstractProjectionRegularization} = norm(R, x)
λ(::AbstractProjectionRegularization) = nothing

export AbstractRegularizationNormalization, NoNormalization, MeasurementBasedNormalization, SystemMatrixBasedNormalization
abstract type AbstractRegularizationNormalization end
struct NoNormalization <: AbstractRegularizationNormalization end
struct MeasurementBasedNormalization <: AbstractRegularizationNormalization end
struct SystemMatrixBasedNormalization <: AbstractRegularizationNormalization end
# TODO weighted systemmatrix, maybe weighted measurementbased?

struct NormalizedRegularization{TF, T, R<:AbstractParameterizedRegularization{T}} <: AbstractParameterizedRegularization{TF}
  reg::R
  factor::TF
end
λ(reg::NormalizedRegularization) = λ(reg.reg) * reg.factor
prox!(reg::NormalizedRegularization, x, λ) = prox!(reg.reg, x, λ)
norm(reg::NormalizedRegularization, x, λ) = norm(reg.reg, x, λ)

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

normalize(solver::AbstractLinearSolver, norm, regs, A, b) = normalize(typeof(solver), norm, regs, A, b)
normalize(solver::Type{T}, norm::AbstractRegularizationNormalization, regs, A, b) where T<:AbstractLinearSolver = normalize(norm, regs, A, b)
# System matrix based normalization is already done in constructor, can just return regs
normalize(solver::AbstractLinearSolver, norm::SystemMatrixBasedNormalization, regs, A, b) = regs

"""
Type describing custom regularizers

# Fields
* `prox!::Function`           - proximal map for the regularizer
* `norm::Function`            - (semi-)norm for the regularizer
* `λ::AbstractFloat`                - regularization paramter
* `params::Dict{Symbol,Any}`  - additional parameters
"""
mutable struct CustomRegularization{T} <: AbstractParameterizedRegularization{T}
  prox!::Function
  norm::Function
  λ::T
  params::Dict{Symbol,Any}  # @TODO in die funcs
end

prox!(reg::CustomRegularization, x) = reg.prox!(x, reg.λ; reg.params...)
norm(reg::CustomRegularization, x) = reg.norm(x, reg.λ; reg.params...)
Regularization(prox!::Function = x->x, norm::Function = norm0, λ::AbstractFloat=0.0, params=Dict{Symbol,Any}()) = CustomRegularization(prox!, norm, λ, params)

Base.vec(reg::AbstractRegularization) = AbstractRegularization[reg]
Base.vec(reg::Vector{AbstractRegularization}) = reg

"""
    RegularizationList()

Returns a list of all available Regularizations
"""
function RegularizationList()
  Any["L2", "L1", "L21", "TV", "LLR", "Positive", "Proj", "Nuclear"]
end

"""
    Regularization(name::String, λ::AbstractFloat; kargs...)

create a Regularization object containing all the infos necessary to calculate a proximal map.

# valid names
* `"L2"`
* `"L1"`
* `"L21"`
* `"TV"`
* `"LLR"`
* `"Nuclear"`
* `"Positive"`
* `"Proj"`
"""
function Regularization(name::String, λ::AbstractFloat; kargs...)
  if name=="L2"
    return L2Regularization(λ)
  elseif name=="L1"
    return L1Regularization(λ; kargs...)
  elseif name=="L21"
    return L21Regularization(λ; kargs...)
  elseif name=="TV"
    # preallocate fields for computation of proximal map
    # shape = get(kargs, :shape, nothing)
    # if haskey(kargs, :T)
    #   T = kargs[:T]
    # else
    #   @info "no type T for TV-regularization given. Assuming ComplexF64"
    #   T = ComplexF64
    # end
    # tvpar = TVParams(shape, T; kargs...)
    # tvprox! = (x,λ)->proxTV!(x,λ,tvpar; kargs...)
    # return Regularization(tvprox!, normTV, λ, kargs)
    return TVRegularization(λ; kargs...)
  elseif name=="LLR"
    return LLRRegularization(λ; kargs...)
  elseif name=="Nuclear"
    return NuclearRegularization(λ; kargs...)
  elseif name=="Positive"
    return PositiveRegularization(λ)
  elseif name=="Proj"
    return ProjectionRegularization(λ; kargs...)
  else
    error("Regularization $name not found.")
  end
end

"""
    Regularization(names::Vector{String}, λ::Vector{Float64}; kargs...)

create a Regularization object containing all the infos necessary to calculate a proximal map.
Valid names are the same as in `Regularization(name::String, λ::AbstractFloat; kargs...)`.
"""
function Regularization(names::Vector{String},
                        λ::Vector{T}; kargs...) where T<:AbstractFloat
  #@Mirco I do not know what to do with kargs here
  if length(names) != length(λ)
    @error names " and " λ " need to have the same length "
  end
  reg = []
  for l=1:length(names)
    push!(reg, Regularization(names[l],λ[l]; kargs...))
  end
  return reg
end

norm0(x::Array{T}, λ::T; sparseTrafo::Trafo=nothing, kargs...) where T = 0.0
