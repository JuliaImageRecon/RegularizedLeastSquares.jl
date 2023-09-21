export MaskedRegularization

struct MaskedRegularization{R<:AbstractRegularization} <: AbstractRegularization
  reg::R
  constraintMask::Vector{Bool}
end
λ(reg::MaskedRegularization) = λ(reg.reg)
nested(reg::MaskedRegularization) = reg.reg


function prox!(reg::MaskedRegularization, x::AbstractArray)
	z = view(x, findall(reg.constraintMask))
  prox!(reg.reg, z)
	return x
end
function norm(reg::MaskedRegularization, x::AbstractArray)
  z = view(x, findall(reg.constraintMask))
  result = norm(reg.reg, z)
  return result
end