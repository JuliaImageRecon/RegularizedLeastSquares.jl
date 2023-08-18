export SparseRegularization

struct SparseRegularization{T, R<:AbstractRegularization} <: AbstractRegularization
  sparseTrafo::Trafo
  reg::R
end


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