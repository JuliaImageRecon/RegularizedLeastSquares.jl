export AbstractRegularization, AbstractParameterizedRegularization, AbstractProjectionRegularization, prox!, nested, sink, sinktype, λ, findsink, findsinks

abstract type AbstractRegularization end
inner(::AbstractRegularization) = nothing
iterate(reg::AbstractRegularization, state = reg) = isnothing(state) ? nothing : (state, inner(state))
Base.IteratorSize(::AbstractRegularization) = Base.SizeUnknown()
sink(reg::AbstractRegularization) = reg
sinktype(reg::AbstractRegularization) = typeof(sink(reg))


abstract type AbstractParameterizedRegularization{T} <: AbstractRegularization end
"""
    prox!(reg::AbstractParameterizedRegularization, x)

perform the proximal mapping defined by `reg` on `x`. Uses the regularization parameter defined for `reg`.
"""
prox!(reg::AbstractParameterizedRegularization, x::AbstractArray) = prox!(reg, x, λ(reg))
"""
    norm(reg::AbstractParameterizedRegularization, x)

returns the value of the `reg` regularization term on `x`. Uses the regularization parameter defined for `reg`.
"""
norm(reg::AbstractParameterizedRegularization, x::AbstractArray) = norm(reg, x, λ(reg))
"""
    λ(reg::AbstractParameterizedRegularization)

return the regularization parameter `λ` of `reg`
"""
λ(reg::AbstractParameterizedRegularization) = reg.λ
# Conversion
prox!(reg::AbstractParameterizedRegularization, x::AbstractArray{Tc}, λ) where {T, Tc<:Union{T, Complex{T}}} = prox!(reg, x, convert(T, λ))
norm(reg::AbstractParameterizedRegularization, x::AbstractArray{Tc}, λ) where {T, Tc<:Union{T, Complex{T}}} = norm(reg, x, convert(T, λ))

"""
    prox!(regType::Type{<:AbstractParameterizedRegularization}, x, λ; kwargs...)
  
construct a regularization term of type `regType` with given `λ` and `kwargs` and apply its `prox!` on `x`
"""
prox!(regType::Type{<:AbstractParameterizedRegularization}, x, λ; kwargs...) = prox!(regType(λ; kwargs...), x, λ)
"""
    norm(regType::Type{<:AbstractParameterizedRegularization}, x, λ; kwargs...)
  
construct a regularization term of type `regType` with given `λ` and `kwargs` and apply its `norm` on `x`
"""
norm(regType::Type{<:AbstractParameterizedRegularization}, x, λ; kwargs...) = norm(regType(λ; kwargs...), x, λ)

abstract type AbstractProjectionRegularization <: AbstractRegularization end
λ(::AbstractProjectionRegularization) = nothing

"""
    prox!(regType::Type{<:AbstractProjectionRegularization}, x; kwargs...)
  
construct a regularization term of type `regType` with given `kwargs` and apply its `prox!` on `x`
"""
prox!(regType::Type{<:AbstractProjectionRegularization}, x; kwargs...) = prox!(regType(;kwargs...), x)
"""
    norm(regType::Type{<:AbstractProjectionRegularization}, x; kwargs...)
  
construct a regularization term of type `regType` with given `kwargs` and apply its `norm` on `x`
"""
norm(regType::Type{<:AbstractProjectionRegularization}, x; kwargs...) = norm(regType(;kwargs...), x)

include("NestedRegularization.jl")
include("ScaledRegularization.jl")
include("NormalizedRegularization.jl")
include("TransformedRegularization.jl")
include("MaskedRegularization.jl")
include("ConstraintTransformedRegularization.jl")
include("PlugAndPlayRegularization.jl")


function findfirst(::Type{S}, reg::AbstractRegularization) where S <: AbstractRegularization 
  regs = collect(reg)
  idx = findfirst(x->x isa S, regs)
  isnothing(idx) ? nothing : regs[idx]
end
function findsink(::Type{S}, reg::Vector{<:AbstractRegularization}) where S <: AbstractRegularization
  all = findall(x -> sinktype(x) <: S, reg)
  if isempty(all)
    return nothing
  elseif length(all) == 1
    return first(all)
  else
    error("Cannot unambigiously retrieve reg term of type $S, found $(length(all)) instances")
  end
end

findsinks(::Type{S}, reg::Vector{<:AbstractRegularization}) where S <: AbstractRegularization = findall(x -> sinktype(x) <: S, reg)


Base.vec(reg::AbstractRegularization) = AbstractRegularization[reg]
Base.vec(reg::Vector{AbstractRegularization}) = reg

"""
    RegularizationList()

Returns a list of all available Regularizations
"""
function RegularizationList()
  return subtypes(RegularizationList) # TODO loop over abstract types and push! to list
end

norm0(x::Array{T}, λ::T; sparseTrafo=nothing, kargs...) where T = 0.0
