export ConstraintTransformedRegularization, transform

"""
    ConstraintTransformedRegularization

Nested regularization term that associates the `nested` regularization term with a transform.
"""
struct ConstraintTransformedRegularization{S, R<:AbstractRegularization, TR} <: AbstractNestedRegularization{S}
  reg::R
  trafo::TR
  ConstraintTransformedRegularization(reg::AbstractRegularization, trafo::TR) where TR = new{R, R, TR}(reg, trafo)
  ConstraintTransformedRegularization(reg::R, trafo::TR) where {S, R<:AbstractNestedRegularization{S}, TR} = new{S,R, TR}(reg, trafo)
end
inner(reg::ConstraintTransformedRegularization) = reg.reg
"""
    transform(reg::ConstraintTransformedRegularization)

return the transform associated with `inner(reg)`.
"""
transform(reg::ConstraintTransformedRegularization) = reg.trafo