export TransformedRegularization

struct TransformedRegularization{R<:AbstractRegularization, S} <: AbstractRegularization
  reg::R
  sparseTrafo::S
end
sink(reg::TransformedRegularization) = sink(reg.reg)
λ(reg::TransformedRegularization) = λ(reg.reg)

function prox!(reg::TransformedRegularization, x::AbstractArray)
	z = reg.sparseTrafo * x
  result = prox!(reg.reg, z)
	x[:] = adjoint(reg.sparseTrafo) * result
  return x
end
function norm(reg::TransformedRegularization, x::AbstractArray)
  z = reg.sparseTrafo * x 
  result = norm(reg.reg, z)
  return result
end

function prox!(reg::TransformedRegularization, x::AbstractArray, λ)
	z = reg.sparseTrafo * x
  result = prox!(reg.reg, z, λ)
	x[:] = adjoint(reg.sparseTrafo) * result
  return x
end
function norm(reg::TransformedRegularization, x::AbstractArray, λ)
  z = reg.sparseTrafo * x 
  result = norm(reg.reg, z, λ)
  return result
end