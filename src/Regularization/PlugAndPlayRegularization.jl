
export PnPRegularization, PlugAndPlayRegularization
"""
        PlugAndPlayRegularization

Regularization term implementing a given plug-and-play proximal mapping.
The actual regularization term is indirectly defined by the learned proximal mapping and as such there is no `norm` implemented.

# Arguments
* `λ`                  - regularization paramter

# Keywords
* `model`       - model applied to the image
* `input_transform` - transform of image before `model`
"""
struct PlugAndPlayRegularization{T, M, I} <: AbstractParameterizedRegularization{T}
    model::M
    λ::T
    input_transform::I
    ignoreIm::Bool
    PlugAndPlayRegularization(λ::T; model::M, input_transform::I=RegularizedLeastSquares.MinMaxTransform, ignoreIm = false, kargs...) where {T<:Number, M, I} = new{T, M, I}(model, λ, input_transform, ignoreIm)
end
PlugAndPlayRegularization(model; kwargs...) = PlugAndPlayRegularization(one(Float32); kwargs..., model = model)

function prox!(self::PlugAndPlayRegularization, x::AbstractArray{Tc}, λ::T) where {T, Tc <: Complex{T}}
    out = real.(x)
    if self.ignoreIm
        x[:] = prox!(self, out, λ) + imag(x) * imag(one(T))
    else
        x[:] = prox!(self, real.(x), λ) + prox!(self, imag.(x), λ) * imag(one(T))
    end
    return x
end

function prox!(self::PlugAndPlayRegularization, x::AbstractArray{T}, λ::T) where {T}

    if λ != self.λ && (λ < 0.0 || λ > 1.0)
        temp = clamp(λ, zero(T), one(T))
        @warn "$(typeof(self)) was given λ with value $λ. Valid range is [0, 1]. λ changed to temp"
        λ = temp
      end

    out = copy(x)
    tf = self.input_transform(out)

    out = RegularizedLeastSquares.transform(tf, out)
    out = out - λ * (out - self.model(out))
    out = RegularizedLeastSquares.inverse_transform(tf, out)

    x[:] = vec(out)
    return x
end

PnPRegularization = PlugAndPlayRegularization