abstract type AbstractScaledRegularization{T} <: AbstractParameterizedRegularization{T} end

export FixedScaledRegularization
struct FixedScaledRegularization{T, R<:AbstractParameterizedRegularization{T}} <: AbstractScaledRegularization{T}
  reg::R
  factor::T
end

prox!(reg::FixedScaledRegularization{T}, x::AbstractArray{Tc}; factor = one(T)) where {T, Tc <: Union{T, Complex{T}}} = prox!(reg.reg, x; factor =  reg.factor / factor)
prox!(reg::FixedScaledRegularization, x::AbstractArray{T}; factor = one(T)) where {T} = prox!(reg.reg, x; factor =  reg.factor / factor)
norm(reg::FixedScaledRegularization{T}, x::AbstractArray{Tc}; factor = one(T)) where {T, Tc <: Union{T, Complex{T}}} = norm(reg.reg, x; factor =  reg.factor / factor)
norm(reg::FixedScaledRegularization, x::AbstractArray{T}; factor = one(T)) where {T} = norm(reg.reg, x; factor =  reg.factor / factor)

export AutoScaledRegularization
mutable struct AutoScaledRegularization{T, R<:AbstractParameterizedRegularization{T}} <: AbstractScaledRegularization{T}
  reg::R
  factor::Union{Nothing, T}
  AutoScaledRegularization(reg::R) where {T, R<:AbstractParameterizedRegularization{T}} = new{T,R}(reg, nothing)
end

initFactor!(reg::AutoScaledRegularization, x::AbstractArray) = reg.factor = maximum(abs.(x))
function prox!(reg::AutoScaledRegularization{T}, x::AbstractArray{Tc}; factor = one(T)) where {T, Tc <: Union{T, Complex{T}}}
  isnothing(reg.factor) && initFactor!(reg, x)
  prox!(reg.reg, x; factor = reg.factor / factor)
end
function prox!(reg::AutoScaledRegularization, x::AbstractArray{T}; factor = one(T)) where {T}
  isnothing(reg.factor) && initFactor!(reg, x)
  prox!(reg.reg, x; factor = reg.factor / factor)
end

function norm(reg::AutoScaledRegularization{T}, x::AbstractArray{Tc}; factor = one(T)) where {T, Tc <: Union{T, Complex{T}}}
  isnothing(reg.factor) && initFactor!(reg, x)
  norm(reg.reg, x; factor = reg.factor / factor)
end
function norm(reg::AutoScaledRegularization, x::AbstractArray{T}; factor = one(T)) where {T}
  isnothing(reg.factor) && initFactor!(reg, x)
  norm(reg.reg, x; factor = reg.factor / factor)
end