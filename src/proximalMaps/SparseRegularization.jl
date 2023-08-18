export SparseRegularization

struct SparseRegularization{T, R<:AbstractRegularization} <: AbstractRegularization
  sparseTrafo::Trafo
  reg::R
end
λ(reg::SparseRegularization) = λ(reg.reg)

function prox!(reg::SparseRegularization, x::AbstractArray; kwargs...)
	z = reg.sparseTrafo * x
  result = prox!(reg.reg, z; kwargs...)
	return adjoint(reg.sparseTrafo) * result
end
function norm(reg::SparseRegularization, x::AbstractArray; kwargs...)
  z = reg.sparseTrafo * x 
  result = norm(reg.reg, z; kwargs...)
  return result
end