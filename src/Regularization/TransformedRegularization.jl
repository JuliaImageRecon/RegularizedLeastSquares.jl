export TransformedRegularization

"""
    TransformedRegularization(reg, trafo)

Nested regularization term that applies `prox!` or `norm` on `z = trafo * x` and returns (inplace) the `adjoint(trafo)` of the result. 
"""
struct TransformedRegularization{S, R<:AbstractRegularization, TR} <: AbstractNestedRegularization{S}
  reg::R
  trafo::TR
  TransformedRegularization(reg::R, trafo::TR) where {R<:AbstractRegularization, TR} = new{R, R, TR}(reg, trafo)
  TransformedRegularization(reg::R, trafo::TR) where {S, R<:AbstractNestedRegularization{S}, TR} = new{S,R, TR}(reg, trafo)
end
inner(reg::TransformedRegularization) = reg.reg

function prox!(reg::TransformedRegularization, x::AbstractArray, args...)
	z = reg.trafo * x
  result = prox!(reg.reg, z, args...)
	x[:] = adjoint(reg.trafo) * result
  return x
end
function norm(reg::TransformedRegularization, x::AbstractArray, args...)
  z = reg.trafo * x 
  result = norm(reg.reg, z, args...)
  return result
end