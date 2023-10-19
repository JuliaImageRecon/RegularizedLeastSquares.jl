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