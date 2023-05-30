export Regularization, AbstractRegularization, lambdList, prox! #, norm

abstract type AbstractRegularization{T} end
prox!(reg::AbstractRegularization{T}, x::AbstractArray{Tc}; factor::T = 1) where {T, Tc <: Union{T, Complex{T}}} = prox!(reg, x, reg.λ * factor)
norm(reg::AbstractRegularization{T}, x::AbstractArray{Tc}; factor::T = 1) where {T, Tc <: Union{T, Complex{T}}} = norm(reg, x, reg.λ * factor)
prox!(reg::AbstractRegularization, x::AbstractArray{T}; factor::T = 1) where {T} = prox!(reg, x, T(reg.λ * factor))
norm(reg::AbstractRegularization, x::AbstractArray{T}; factor::T = 1) where {T} = norm(reg, x, T(reg.λ * factor))

@generated function prox!(reg::T, x, λ) where {T<:AbstractRegularization}
  kwargs = [Expr(:kw, :($field), :(reg.$field)) for field in fieldnames(T)[2:end]]
  return Expr(:call, :prox!, Expr(:parameters, kwargs...), T, :x, :λ)
end

@generated function norm(reg::T, x, λ) where {T<:AbstractRegularization}
  kwargs = [Expr(:kw, :($field), :(reg.$field)) for field in fieldnames(T)[2:end]]
  return Expr(:call, :norm, Expr(:parameters, kwargs...), T, :x, :λ)
end

export AbstractRegularizationNormalization, NoNormalization, MeasurementBasedNormalization, SystemMatrixBasedNormalization
abstract type AbstractRegularizationNormalization end
struct NoNormalization <: AbstractRegularizationNormalization end
struct MeasurementBasedNormalization <: AbstractRegularizationNormalization end
struct SystemMatrixBasedNormalization <: AbstractRegularizationNormalization end
# TODO weighted systemmatrix, maybe weighted measurementbased?

function normalize(::MeasurementBasedNormalization, regs, A, b::AbstractArray)
  return norm(b, 1)/length(b)
end
normalize(::MeasurementBasedNormalization, regs, A, b::Nothing) = normalize(NoNormalization(), regs, A, b)
function normalize(::SystemMatrixBasedNormalization, regs, A::AbstractArray, b)
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
normalize(::NoNormalization, regs, A, b) = 1.0


normalize(solver::AbstractLinearSolver, norm::AbstractRegularizationNormalization, regs, A, b) = normalize(norm, regs, A, b)
# System matrix based normalization is already done in constructor
normalize(solver::AbstractLinearSolver, norm::SystemMatrixBasedNormalization, regs, A, b) = solver.regFac

"""
Type describing custom regularizers

# Fields
* `prox!::Function`           - proximal map for the regularizer
* `norm::Function`            - (semi-)norm for the regularizer
* `λ::AbstractFloat`                - regularization paramter
* `params::Dict{Symbol,Any}`  - additional parameters
"""
mutable struct CustomRegularization{T} <: AbstractRegularization{T}
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
