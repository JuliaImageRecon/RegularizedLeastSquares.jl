
export PnPRegularization, PlugAndPlayRegularization
"""
        PlugAndPlayRegularization

Regularization term implementing a given plug-and-play proximal mapping.
The actual regularization term is indirectly defined by the learned proximal mapping and as such there is no `norm` implemented.

# Arguments
* `λ`                  - regularization paramter

# Keywords
* `model`       - model applied to the image
* `shape`       - dimensions of the image
* `input_transform` - transform of image before `model`
"""
struct PlugAndPlayRegularization{T, M, I} <: AbstractParameterizedRegularization{T}
    model::M
    λ::T
    shape::Vector{Int}
    input_transform::I
    ignoreIm::Bool
    PlugAndPlayRegularization(λ::T; model::M, shape, input_transform::I=RegularizedLeastSquares.MinMaxTransform, ignoreIm = false, kargs...) where {T, M, I} = new{T, M, I}(model, λ, shape, input_transform, ignoreIm)
end
PlugAndPlayRegularization(model, shape; kwargs...) = PlugAndPlayRegularization(one(Float32); kwargs..., model = model, shape = shape)

function prox!(self::PlugAndPlayRegularization, x::AbstractArray{Complex{T}}, λ::T) where {T <: Real}
    if self.ignoreIm
        copyto!(x, prox!(self, real.(x), λ) + imag.(x) * one(T)im)
    else
        copyto!(x, prox!(self, real.(x), λ) + prox!(self, imag.(x), λ) * one(T)im)
    end
    return x
end

function prox!(self::PlugAndPlayRegularization, x::AbstractArray{T}, λ::T) where {T <: Real}

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

    copyto!(x, vec(out))
    return x
end

PnPRegularization = PlugAndPlayRegularization