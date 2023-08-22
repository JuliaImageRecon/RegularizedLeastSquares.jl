export ConstrainedRegularization

struct ConstrainedRegularization{R<:AbstractRegularization} <: AbstractRegularization
  reg::R
  constraintMask::Vector{Bool}
end
λ(reg::ConstrainedRegularization) = λ(reg.reg)
sink(reg::ConstrainedRegularization) = sink(reg.reg)


function prox!(reg::ConstrainedRegularization, x::AbstractArray)
	z = view(x, findall(reg.constraintMask))
  prox!(reg.reg, z)
	return x
end
function norm(reg::ConstrainedRegularization, x::AbstractArray)
  z = view(x, findall(reg.constraintMask))
  result = norm(reg.reg, z)
  return result
end