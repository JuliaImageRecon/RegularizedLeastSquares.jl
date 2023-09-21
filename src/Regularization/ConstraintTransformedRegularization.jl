export ConstraintTransformedRegularization, transform

struct ConstraintTransformedRegularization{R<:AbstractRegularization, TR} <: AbstractRegularization
  reg::R
  trafo::TR
end
nested(reg::ConstraintTransformedRegularization) = reg.reg
λ(reg::ConstraintTransformedRegularization) = λ(reg.reg)
transform(reg::ConstraintTransformedRegularization) = reg.trafo

prox!(reg::ConstraintTransformedRegularization, x::AbstractArray, λ) = prox!(reg.reg, x, λ)
norm(reg::ConstraintTransformedRegularization, x::AbstractArray, λ) = norm(reg.reg, x, λ)