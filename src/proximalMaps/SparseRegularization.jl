export SparseRegularization

struct SparseRegularization{T, R<:AbstractRegularization{T}} <: AbstractRegularization{T}
  sparseTrafo::Trafo
  reg::R
end


function prox!(reg::SparseRegularization{T}, x::AbstractArray; factor = nothing) where {T} 
	z = reg.sparseTrafo * x
	result = isnothing(factor) ? prox!(reg.reg, z) : prox!(reg.reg, z; factor = factor)
	return adjoint(reg.sparseTrafo) * result
end
function norm(reg::SparseRegularization{T}, x::AbstractArray; factor = nothing) where {T}
  z = reg.sparseTrafo * x 
  result = isnothing(factor) ? norm(reg.reg, z) : norm(reg, z; factor = factor)
  return result
end