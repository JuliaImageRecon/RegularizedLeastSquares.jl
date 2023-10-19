struct PnPRegularization{T} <: AbstractParameterizedRegularization{T}
    flux_model
    λ::T
    shape::Vector{Int}
    input_transform
    PnPRegularization(model, λ::T, shape; input_transform=MinMaxTransform, kargs...) where T = new{T}(model, λ, shape, input_transform)
end
PnPRegularization(model, shape) = PnPRegularization(model, Float32(1.0), shape)

# complex are not supported
function RegularizedLeastSquares.prox!(self::PnPRegularization, x::AbstractArray{Tc}, λ::T) where {T, Tc <: Union{T, Complex{T}}}

    out = copy(x)
    out = real.(reshape(out, self.shape...))

    tf = self.input_transform(out)

    out = transform(tf, out)
    out = out - λ * (out - self.flux_model(out))
    out = inverse_transform(tf, out)

    x[:] = reshape(out, size(x)...) + imag(x) * 1.0im
end
