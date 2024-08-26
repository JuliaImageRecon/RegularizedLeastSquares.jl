export TransformedRegularization

"""
    TransformedRegularization(reg, trafo)

Nested regularization term that applies `prox!` or `norm` on `z = trafo * x` and returns (inplace) `x = adjoint(trafo) * z`.

# Example
```julia
julia> core = L1Regularization(0.8)
L1Regularization{Float64}(0.8)

julia> wop = WaveletOp(Float32, shape = (32,32));

julia> reg = TransformedRegularization(core, wop);

julia> prox!(reg, randn(32*32)); # Apply soft-thresholding in Wavelet domain
```
"""
struct TransformedRegularization{S, R<:AbstractRegularization, TR} <: AbstractNestedRegularization{S}
  reg::R
  trafo::TR
  TransformedRegularization(reg::R, trafo::TR) where {R<:AbstractRegularization, TR} = new{R, R, TR}(reg, trafo)
  TransformedRegularization(reg::R, trafo::TR) where {S, R<:AbstractNestedRegularization{S}, TR} = new{S,R, TR}(reg, trafo)
end
innerreg(reg::TransformedRegularization) = reg.reg

function prox!(reg::TransformedRegularization, x::AbstractArray, args...)
	z = reg.trafo * x
  result = prox!(reg.reg, z, args...)
	copyto!(x, adjoint(reg.trafo) * result)
  return x
end
function norm(reg::TransformedRegularization, x::AbstractArray, args...)
  z = reg.trafo * x 
  result = norm(reg.reg, z, args...)
  return result
end