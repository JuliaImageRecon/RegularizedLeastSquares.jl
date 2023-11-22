export AbstractNestedRegularization

abstract type AbstractNestedRegularization{S} <: AbstractRegularization end
"""
    innerreg(reg::AbstractNestedRegularization)

return the `inner` regularization term of `reg`. Nested regularization terms also implement the iteration interface.
"""
innerreg(::R) where R<:AbstractNestedRegularization = error("Nested regularization term $R must implement nested")
"""
    sink(reg::AbstractNestedRegularization)

return the innermost regularization term.
"""
sink(reg::AbstractNestedRegularization{S}) where S = last(collect(reg))
"""
    sinktype(reg::AbstractNestedRegularization)

return the type of the innermost regularization term.

See also [`sink`](@ref).
"""
sinktype(::AbstractNestedRegularization{S}) where S = S
λ(reg::AbstractNestedRegularization) = λ(innerreg(reg))

prox!(reg::AbstractNestedRegularization{S}, x) where S <: AbstractParameterizedRegularization = prox!(reg, x, λ(reg))
norm(reg::AbstractNestedRegularization{S}, x) where S <: AbstractParameterizedRegularization = norm(reg, x, λ(reg))

prox!(reg::AbstractNestedRegularization, x, args...) = prox!(innerreg(reg), x, args...)
norm(reg::AbstractNestedRegularization, x, args...) = norm(innerreg(reg), x, args...)