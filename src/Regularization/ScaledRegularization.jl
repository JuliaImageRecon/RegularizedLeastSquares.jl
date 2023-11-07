export AbstractScaledRegularization, factor
#See also [factor](@ref), [λ](@ref), [nested](@ref).
"""
    AbstractScaledRegularization

Nested regularization term that applies a `factor` to the regularization parameter `λ` of its `nested` term.

"""
abstract type AbstractScaledRegularization{T, S<:AbstractParameterizedRegularization{T}} <: AbstractNestedRegularization{S} end
"""
    factor(reg::AbstractScaledRegularization)

return the scaling `factor` for `λ`
"""
factor(::R) where R <: AbstractScaledRegularization = error("Scaled regularization term $R must implement factor")
#See also [factor](@ref), [λ](@ref), [nested](@ref).
"""
    λ(reg::AbstractScaledRegularization)

return `λ` of `nested` regularization term scaled by `factor(reg)`.

"""
λ(reg::AbstractScaledRegularization) = λ(nested(reg)) * factor(reg)

export FixedScaledRegularization
struct FixedScaledRegularization{T, S, R} <: AbstractScaledRegularization{T, S}
  reg::R
  factor::T
  FixedScaledRegularization(reg::R, factor) where {T, R <: AbstractParameterizedRegularization{T}} = new{T, R, R}(reg, factor)
  FixedScaledRegularization(reg::R, factor) where {T, RN <: AbstractParameterizedRegularization{T}, R<:AbstractNestedRegularization{RN}} = new{T, RN, R}(reg, factor)
end
nested(reg::FixedScaledRegularization) = reg.reg
factor(reg::FixedScaledRegularization) = reg.factor

export FixedParameterRegularization
struct FixedParameterRegularization{T, S, R} <: AbstractScaledRegularization{T, S}
  reg::R
  FixedParameterRegularization(reg::R) where {T, R <: AbstractParameterizedRegularization{T}} = new{T, R, R}(reg)
  FixedScaledRegularization(reg::R) where {T, RN <: AbstractParameterizedRegularization{T}, R<:AbstractNestedRegularization{RN}} = new{T, RN, R}(reg)
end
factor(reg::FixedParameterRegularization) = 1.0
nested(reg::FixedParameterRegularization) = reg.reg
# Drop any incoming λ and subsitute inner
prox!(reg::FixedParameterRegularization, x, discard) = prox!(nested(reg), x, λ(nested(reg)))
norm(reg::FixedParameterRegularization, x, discard) = norm(nested(reg), x, λ(nested(reg)))

export AutoScaledRegularization
mutable struct AutoScaledRegularization{T, S, R} <: AbstractScaledRegularization{T, S}
  reg::R
  factor::Union{Nothing, T}
  AutoScaledRegularization(reg::R) where {T, R <: AbstractParameterizedRegularization{T}} = new{T, R, R}(reg, nothing)
  AutoScaledRegularization(reg::R) where {T, RN <: AbstractParameterizedRegularization{T}, R<:AbstractNestedRegularization{RN}} = new{T, RN, R}(reg, nothing)
end
initFactor!(reg::AutoScaledRegularization, x::AbstractArray) = reg.factor = maximum(abs.(x))
nested(reg::AutoScaledRegularization) = reg.reg
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