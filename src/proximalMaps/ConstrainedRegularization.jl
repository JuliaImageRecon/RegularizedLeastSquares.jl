export ConstrainedRegularization

struct ConstrainedRegularization{T, R<:AbstractRegularization} <: AbstractRegularization
  constraintMask::Vector{Bool}
  reg::R
end
λ(reg::ConstrainedRegularization) = λ(reg.reg)

function prox!(reg::ConstrainedRegularization, x::AbstractArray; kwargs...)
	z = view(x, findall(reg.constraintMask))
  result = prox!(reg.reg, z; kwargs...)
	return result
end
function norm(reg::ConstrainedRegularization, x::AbstractArray; kwargs...)
  z = view(x, findall(reg.constraintMask))
  result = norm(reg.reg, z; kwargs...)
  return result
end