export MaskedRegularization

"""
    MaskedRegularization

Nested regularization term that only applies `prox!` and `norm` to elements of `x` for which the mask is `true`.

# Examples
```julia
julia> positive = PositiveRegularization();

julia> masked = MaskedRegularization(reg, [true, false, true, false]);

julia> prox!(masked, fill(-1, 4))
4-element Vector{Float64}:
  0.0
 -1.0
  0.0
 -1.0
```
"""
struct MaskedRegularization{S, R<:AbstractRegularization} <: AbstractNestedRegularization{S}
  reg::R
  mask::Vector{Bool}
  MaskedRegularization(reg::R, mask) where R <: AbstractRegularization = new{R, R}(reg, mask)
  MaskedRegularization(reg::R, mask) where {S, R<:AbstractNestedRegularization{S}} = new{S,R}(reg, mask)
end
inner(reg::MaskedRegularization) = reg.reg


function prox!(reg::MaskedRegularization, x::AbstractArray, args...)
	z = view(x, findall(reg.mask))
  prox!(reg.reg, z, args...)
	return x
end
function norm(reg::MaskedRegularization, x::AbstractArray, args...)
  z = view(x, findall(reg.mask))
  result = norm(reg.reg, z, args...)
  return result
end