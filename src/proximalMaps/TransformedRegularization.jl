export TransformedRegularization

struct TransformedRegularization{R<:AbstractRegularization, TR} <: AbstractRegularization
  reg::R
  trafo::TR
end
nested(reg::TransformedRegularization) = reg.reg
λ(reg::TransformedRegularization) = λ(reg.reg)

function prox!(reg::TransformedRegularization, x::AbstractArray)
	z = reg.trafo * x
  result = prox!(reg.reg, z)
	x[:] = adjoint(reg.trafo) * result
  return x
end
function norm(reg::TransformedRegularization, x::AbstractArray)
  z = reg.trafo * x 
  result = norm(reg.reg, z)
  return result
end

function prox!(reg::TransformedRegularization, x::AbstractArray, λ)
	z = reg.trafo * x
  result = prox!(reg.reg, z, λ)
	x[:] = adjoint(reg.trafo) * result
  return x
end
function norm(reg::TransformedRegularization, x::AbstractArray, λ)
  z = reg.trafo * x 
  result = norm(reg.reg, z, λ)
  return result
end