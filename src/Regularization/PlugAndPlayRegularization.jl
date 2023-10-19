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

# function RegularizedLeastSquares.prox!(self::PnPRegularization, x::AbstractArray{Tc}, λ::T) where {T, Tc <: Union{T, Complex{T}}}
#     println("x_before", "max=", maximum(real(x)), "; min=", minimum(real(x)), "; mean=", mean(real(x)))
#     out = copy(x)
#     out = real.(reshape(out, self.shape...))
#     min_val = minimum(out)
#     max_val = maximum(out)
#     out = (out .- min_val) ./ (max_val - min_val) #.* 0.7 .- 0.2
#     println("x: ", "max=", maximum(out), "; min=", minimum(out), "; mean=", mean(out))
#     println("mean(out - self.model(out)) ", mean(out - self.flux_model(out)))
#     out = out - λ * (out - self.flux_model(out))
#     # out = min_val .+ ((out .+ 0.2) ./ 0.7 .* (max_val - min_val))
#     out = min_val .+ (out .* (max_val - min_val))
#     x[:] = reshape(out, size(x)...) + imag(x) * 1.0im
#     println("x_after", "max=", maximum(real(x)), "; min=", minimum(real(x)), "; mean=", mean(real(x)))
#   end