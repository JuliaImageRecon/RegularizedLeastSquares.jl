export AbstractNestedRegularization

abstract type AbstractNestedRegularization{S} <: AbstractRegularization end
"""
    nested(reg::AbstractNestedRegularization)

return the regularization term that `reg`end is decorating. Nested regularization terms also implement the iteration interface.
"""
nested(::R) where R<:AbstractNestedRegularization = error("Nested regularization term $R must implement nested")
"""
    sink(reg::AbstractNestedRegularization)

return the innermost regularization term.
"""
sink(reg::AbstractNestedRegularization{S}) where S = last(collect(reg))
#See also [sink](@ref).
"""
    sinktype(reg::AbstractNestedRegularization)

return the type of the innermost regularization term.

"""
sinktype(::AbstractNestedRegularization{S}) where S = S
λ(reg::AbstractNestedRegularization) = λ(nested(reg))

prox!(reg::AbstractNestedRegularization{S}, x) where S <: AbstractParameterizedRegularization = prox!(reg, x, λ(reg))
norm(reg::AbstractNestedRegularization{S}, x) where S <: AbstractParameterizedRegularization = norm(reg, x, λ(reg))

prox!(reg::AbstractNestedRegularization, x, args...) = prox!(nested(reg), x, args...)
norm(reg::AbstractNestedRegularization, x, args...) = norm(nested(reg), x, args...)