export AbstractScaledRegularization, factor
"""
    AbstractScaledRegularization

Nested regularization term that applies a `factor` to the regularization parameter `λ` of its `inner` term.

See also [`factor`](@ref), [`λ`](@ref), [`inner`](@ref).
"""
abstract type AbstractScaledRegularization{T, S<:AbstractParameterizedRegularization{T}} <: AbstractNestedRegularization{S} end
"""
    factor(reg::AbstractScaledRegularization)

return the scaling `factor` for `λ`
"""
factor(::R) where R <: AbstractScaledRegularization = error("Scaled regularization term $R must implement factor")
"""
    λ(reg::AbstractScaledRegularization)

return `λ` of `inner` regularization term scaled by `factor(reg)`.

See also [`factor`](@ref), [`inner`](@ref).
"""
λ(reg::AbstractScaledRegularization) = λ(inner(reg)) * factor(reg)

export FixedScaledRegularization
struct FixedScaledRegularization{T, S, R} <: AbstractScaledRegularization{T, S}
  reg::R
  factor::T
  FixedScaledRegularization(reg::R, factor) where {T, R <: AbstractParameterizedRegularization{T}} = new{T, R, R}(reg, factor)
  FixedScaledRegularization(reg::R, factor) where {T, RN <: AbstractParameterizedRegularization{T}, R<:AbstractNestedRegularization{RN}} = new{T, RN, R}(reg, factor)
end
inner(reg::FixedScaledRegularization) = reg.reg
factor(reg::FixedScaledRegularization) = reg.factor

export FixedParameterRegularization
"""
    FixedParameterRegularization

Nested regularization term that discards any `λ` passed to it and instead uses `λ` from its inner regularization term. This can be used to selectively disallow normalization. 
"""
struct FixedParameterRegularization{T, S, R} <: AbstractScaledRegularization{T, S}
  reg::R
  FixedParameterRegularization(reg::R) where {T, R <: AbstractParameterizedRegularization{T}} = new{T, R, R}(reg)
  FixedScaledRegularization(reg::R) where {T, RN <: AbstractParameterizedRegularization{T}, R<:AbstractNestedRegularization{RN}} = new{T, RN, R}(reg)
end
factor(reg::FixedParameterRegularization) = 1.0
inner(reg::FixedParameterRegularization) = reg.reg
# Drop any incoming λ and subsitute inner
prox!(reg::FixedParameterRegularization, x, discard) = prox!(inner(reg), x, λ(inner(reg)))
norm(reg::FixedParameterRegularization, x, discard) = norm(inner(reg), x, λ(inner(reg)))

export AutoScaledRegularization
mutable struct AutoScaledRegularization{T, S, R} <: AbstractScaledRegularization{T, S}
  reg::R
  factor::Union{Nothing, T}
  AutoScaledRegularization(reg::R) where {T, R <: AbstractParameterizedRegularization{T}} = new{T, R, R}(reg, nothing)
  AutoScaledRegularization(reg::R) where {T, RN <: AbstractParameterizedRegularization{T}, R<:AbstractNestedRegularization{RN}} = new{T, RN, R}(reg, nothing)
end
initFactor!(reg::AutoScaledRegularization, x::AbstractArray) = reg.factor = maximum(abs.(x))
inner(reg::AutoScaledRegularization) = reg.reg
# A bit hacky: Factor can only be computed once x is seen, therefore hide factor in λ and silently add it in prox!/norm calls
factor(reg::AutoScaledRegularization) = isnothing(reg.factor) ? 1.0 : reg.factor
function prox!(reg::AutoScaledRegularization, x, λ)
  if isnothing(reg.factor) 
    initFactor!(reg, x) 
    return prox!(reg.reg, x, λ * reg.factor)
  else
    return prox!(reg.reg, x, λ)
  end
end
function norm(reg::AutoScaledRegularization, x, λ)
  if isnothing(reg.factor) 
    initFactor!(reg, x) 
    return norm(reg.reg, x, λ * reg.factor)
  else
    return norm(reg.reg, x, λ)
  end
end