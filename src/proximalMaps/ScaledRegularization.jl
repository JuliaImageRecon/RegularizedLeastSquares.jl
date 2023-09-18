abstract type AbstractScaledRegularization{T} <: AbstractParameterizedRegularization{T} end

export FixedScaledRegularization
struct FixedScaledRegularization{T, R<:AbstractParameterizedRegularization{T}} <: AbstractScaledRegularization{T}
  reg::R
  factor::T
end
nested(reg::FixedScaledRegularization) = reg.reg
λ(reg::FixedScaledRegularization) = λ(reg.reg) * reg.factor
prox!(reg::FixedScaledRegularization, x, λ) = prox!(reg.reg, x, λ)
norm(reg::FixedScaledRegularization, x, λ) = norm(reg.reg, x, λ)


export AutoScaledRegularization
mutable struct AutoScaledRegularization{T, R<:AbstractParameterizedRegularization{T}} <: AbstractScaledRegularization{T}
  reg::R
  factor::Union{Nothing, T}
  AutoScaledRegularization(reg::R) where {T, R<:AbstractParameterizedRegularization{T}} = new{T,R}(reg, nothing)
end
initFactor!(reg::AutoScaledRegularization, x::AbstractArray) = reg.factor = maximum(abs.(x))
nested(reg::AutoScaledRegularization) = reg.reg
# A bit hacky: Factor can only be computed once x is seen, therefore hide factor in λ and silently add it in prox!/norm calls
λ(reg::AutoScaledRegularization) = λ(reg.reg)
function prox!(reg::AutoScaledRegularization, x, λ)
  isnothing(reg.factor) && initFactor!(reg, x) 
  return prox!(reg.reg, x, λ * reg.factor)
end
function norm(reg::AutoScaledRegularization, x, λ)
  isnothing(reg.factor) && initFactor!(reg, x) 
  return norm(reg.reg, x, λ * reg.factor)
end