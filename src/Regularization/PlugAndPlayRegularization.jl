# Defined in /ext/RegularizedLeastSquaresFluxExt
export PnPRegularization, PlugAndPlayRegularization
struct PlugAndPlayRegularization{T, M, I} <: AbstractParameterizedRegularization{T}
    model::M
    λ::T
    shape::Vector{Int}
    input_transform::I
    PlugAndPlayRegularization(model::M, λ::T, shape; input_transform::I=RegularizedLeastSquares.MinMaxTransform, kargs...) where {T, M, I} = new{T, M, I}(model, λ, shape, input_transform)
end
PlugAndPlayRegularization(model, shape) = PlugAndPlayRegularization(model, Float32(1.0), shape)

function prox!(self::PlugAndPlayRegularization, x::AbstractArray{Tc}, λ::T) where {T, Tc <: Complex{T}}
    out = real.(x)
    x[:] = prox!(self, out, λ) + imag(x) * 1.0im
end

function prox!(self::PlugAndPlayRegularization, x::AbstractArray{T}, λ::T) where {T}

    if λ != self.λ && (λ < 0.0 || λ > 1.0)
        temp = clamp(λ, zero(T), one(T))
        @warn "$(typeof(self)) was given λ with value $λ. Valid range is [0, 1]. λ changed to temp"
        λ = temp
      end

    out = copy(x)
    out = reshape(out, self.shape...)

    tf = self.input_transform(out)

    out = RegularizedLeastSquares.transform(tf, out)
    out = out - λ * (out - self.model(out))
    out = RegularizedLeastSquares.inverse_transform(tf, out)

    x[:] = vec(out)
end

PnPRegularization = PlugAndPlayRegularization