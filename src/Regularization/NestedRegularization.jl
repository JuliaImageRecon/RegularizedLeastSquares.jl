export AbstractNestedRegularization

abstract type AbstractNestedRegularization{S} <: AbstractRegularization end
nested(::R) where R<:AbstractNestedRegularization = error("Nested regularization term $R must implement nested")
sink(reg::AbstractNestedRegularization{S}) where S = last(collect(reg))
sinktype(::AbstractNestedRegularization{S}) where S = S

prox!(reg::AbstractNestedRegularization{S}, x) where S <: AbstractParameterizedRegularization = prox!(reg, x, λ(reg))
norm(reg::AbstractNestedRegularization{S}, x) where S <: AbstractParameterizedRegularization = norm(reg, x, λ(reg))