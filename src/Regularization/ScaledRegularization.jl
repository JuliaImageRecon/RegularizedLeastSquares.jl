export AbstractScaledRegularization
"""
    AbstractScaledRegularization

Nested regularization term that applies a `scalefactor` to the regularization parameter `λ` of its `inner` term.

See also [`scalefactor`](@ref), [`λ`](@ref), [`innerreg`](@ref).
"""
abstract type AbstractScaledRegularization{T, S<:AbstractParameterizedRegularization{T}} <: AbstractNestedRegularization{S} end
"""
    scalescalefactor(reg::AbstractScaledRegularization)

return the scaling `scalefactor` for `λ`
"""
scalefactor(::R) where R <: AbstractScaledRegularization = error("Scaled regularization term $R must implement scalefactor")
"""
    λ(reg::AbstractScaledRegularization)

return `λ` of `inner` regularization term scaled by `scalefactor(reg)`.

See also [`scalefactor`](@ref), [`innerreg`](@ref).
"""
λ(reg::AbstractScaledRegularization) = λ(innerreg(reg)) .* scalefactor(reg)

export FixedScaledRegularization
struct FixedScaledRegularization{T, S, R} <: AbstractScaledRegularization{T, S}
  reg::R
  factor::T
  FixedScaledRegularization(reg::R, factor) where {T, R <: AbstractParameterizedRegularization{T}} = new{T, R, R}(reg, factor)
  FixedScaledRegularization(reg::R, factor) where {T, RN <: AbstractParameterizedRegularization{T}, R<:AbstractNestedRegularization{RN}} = new{T, RN, R}(reg, factor)
end
innerreg(reg::FixedScaledRegularization) = reg.reg
scalefactor(reg::FixedScaledRegularization) = reg.factor

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
scalefactor(reg::FixedParameterRegularization) = 1.0
innerreg(reg::FixedParameterRegularization) = reg.reg
# Drop any incoming λ and subsitute inner
prox!(reg::FixedParameterRegularization, x, discard) = prox!(innerreg(reg), x, λ(innerreg(reg)))
norm(reg::FixedParameterRegularization, x, discard) = norm(innerreg(reg), x, λ(innerreg(reg)))

export AutoScaledRegularization
mutable struct AutoScaledRegularization{T, S, R} <: AbstractScaledRegularization{T, S}
  reg::R
  factor::Union{Nothing, T}
  AutoScaledRegularization(reg::R) where {T, R <: AbstractParameterizedRegularization{T}} = new{T, R, R}(reg, nothing)
  AutoScaledRegularization(reg::R) where {T, RN <: AbstractParameterizedRegularization{T}, R<:AbstractNestedRegularization{RN}} = new{T, RN, R}(reg, nothing)
end
initFactor!(reg::AutoScaledRegularization, x::AbstractArray) = reg.factor = maximum(abs.(x))
innerreg(reg::AutoScaledRegularization) = reg.reg
# A bit hacky: Factor can only be computed once x is seen, therefore hide factor in λ and silently add it in prox!/norm calls
scalefactor(reg::AutoScaledRegularization) = isnothing(reg.factor) ? 1.0 : reg.factor
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