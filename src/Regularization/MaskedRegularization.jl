export MaskedRegularization

struct MaskedRegularization{S, R<:AbstractRegularization} <: AbstractNestedRegularization{S}
  reg::R
  constraintMask::Vector{Bool}
  MaskedRegularization(reg::AbstractRegularization, constraintMask) = new{R, R}(reg, constraintMask)
  MaskedRegularization(reg::R, constraintMask) where {S, R<:AbstractNestedRegularization{S}} = new{S,R}(reg, constraintMask)
end
nested(reg::MaskedRegularization) = reg.reg


function prox!(reg::MaskedRegularization, x::AbstractArray, args...)
	z = view(x, findall(reg.constraintMask))
  prox!(reg.reg, z, args...)
	return x
end
function norm(reg::MaskedRegularization, x::AbstractArray, args...)
  z = view(x, findall(reg.constraintMask))
  result = norm(reg.reg, z, args...)
  return result
end