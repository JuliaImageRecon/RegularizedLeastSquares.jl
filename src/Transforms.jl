abstract type AbstractTransform end
# has to implement
# transform(transform::AbstractTransform, x)

abstract type AbstractInveratableTransform <: AbstractTransform end
# has to implement
# inverse_transform(transform::AbstractInveratableTransform, x)


struct MinMaxTransform <: AbstractInveratableTransform
    min
    max
end
MinMaxTransform(x)= MinMaxTransform(minimum(x), maximum(x))

function transform(transform::MinMaxTransform, x)
    return (x .- transform.min) ./ (transform.max .- transform.min)
end

function inverse_transform(transform::MinMaxTransform, x)
    return x .* (transform.max .- transform.min) .+ transform.min
end

struct IdentityTransform <: AbstractInveratableTransform
end
IdentityTransform(x)= IdentityTransform()

function transform(::IdentityTransform, x)
    return x
end

function inverse_transform(::IdentityTransform, x)
    return x
end

struct ZTransform <: AbstractInveratableTransform
    mean
    std
end
ZTransform(x)= ZTransform(mean(x), std(x))

function transform(transform::ZTransform, x)
    return (x .- transform.mean) ./ transform.std
end

function inverse_transform(transform::ZTransform, x)
    return x .* transform.std .+ transform.mean
end

struct ClampedScalingTransform <: AbstractInveratableTransform
    v_min
    v_max
    mask
    x
end
function ClampedScalingTransform(x, v_min, v_max)
    mask = (x .< v_min) .| (x .>= v_max)
    return ClampedScalingTransform(v_min, v_max, mask, x)
end

function transform(transform::ClampedScalingTransform, x)
    return (clamp.(x, transform.v_min, transform.v_max) .- transform.v_min) ./ (transform.v_max - transform.v_min)
end

function inverse_transform(transform::ClampedScalingTransform, x)
    out = x .* (transform.v_max - transform.v_min) .+ transform.v_min
    out[transform.mask] = transform.x[transform.mask]
    return out
end