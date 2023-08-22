export SparseRegularization

struct SparseRegularization{T, R<:AbstractRegularization} <: AbstractRegularization
  reg::R
  sparseTrafo::Trafo
end
λ(reg::SparseRegularization) = λ(reg.reg)

function prox!(reg::SparseRegularization, x::AbstractArray)
	z = reg.sparseTrafo * x
  result = prox!(reg.reg, z)
	return adjoint(reg.sparseTrafo) * result
end
function norm(reg::SparseRegularization, x::AbstractArray)
  z = reg.sparseTrafo * x 
  result = norm(reg.reg, z)
  return result
end