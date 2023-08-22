export ConstrainedRegularization

struct ConstrainedRegularization{T, R<:AbstractRegularization} <: AbstractRegularization
  constraintMask::Vector{Bool}
  reg::R
end
λ(reg::ConstrainedRegularization) = λ(reg.reg)

function prox!(reg::ConstrainedRegularization, x::AbstractArray)
	z = view(x, findall(reg.constraintMask))
  result = prox!(reg.reg, z)
	return result
end
function norm(reg::ConstrainedRegularization, x::AbstractArray)
  z = view(x, findall(reg.constraintMask))
  result = norm(reg.reg, z)
  return result
end