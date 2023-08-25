export SparseRegularization

struct SparseRegularization{R<:AbstractRegularization, S} <: AbstractRegularization
  reg::R
  sparseTrafo::S
end
sink(reg::SparseRegularization) = sink(reg.reg)
λ(reg::SparseRegularization) = λ(reg.reg)

function prox!(reg::SparseRegularization, x::AbstractArray)
	z = reg.sparseTrafo * x
  result = prox!(reg.reg, z)
	x[:] = adjoint(reg.sparseTrafo) * result
  return x
end
function norm(reg::SparseRegularization, x::AbstractArray)
  z = reg.sparseTrafo * x 
  result = norm(reg.reg, z)
  return result
end

function prox!(reg::SparseRegularization, x::AbstractArray, λ)
	z = reg.sparseTrafo * x
  result = prox!(reg.reg, z, λ)
	x[:] = adjoint(reg.sparseTrafo) * result
  return x
end
function norm(reg::SparseRegularization, x::AbstractArray, λ)
  z = reg.sparseTrafo * x 
  result = norm(reg.reg, z, λ)
  return result
end