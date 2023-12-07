export ConstraintTransformedRegularization, transform

"""
    ConstraintTransformedRegularization

Nested regularization term that associates the `nested` regularization term with a transform.

# Arguments
* `reg` - inner regularization term
* `trafo` - transform associated with `reg`
"""
struct ConstraintTransformedRegularization{S, R<:AbstractRegularization, TR} <: AbstractNestedRegularization{S}
  reg::R
  trafo::TR
  ConstraintTransformedRegularization(reg::R, trafo::TR) where {R<:AbstractRegularization, TR} = new{R, R, TR}(reg, trafo)
  ConstraintTransformedRegularization(reg::R, trafo::TR) where {S, R<:AbstractNestedRegularization{S}, TR} = new{S,R, TR}(reg, trafo)
end
innerreg(reg::ConstraintTransformedRegularization) = reg.reg
"""
    transform(reg::ConstraintTransformedRegularization)

return the transform associated with `innerreg(reg)`.
"""
transform(reg::ConstraintTransformedRegularization) = reg.trafo