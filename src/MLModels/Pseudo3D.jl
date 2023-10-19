struct Pseudo3dSliceWise
    model2d
    dim::Integer

    function Pseudo3dSliceWise(
        model2d;
        dim::Integer=1
    )
        new(
            model2d,
            dim,
        )
    end
end


function forward_permutation(self::Pseudo3dSliceWise)::Array{Integer}
    axes = [1, 2, 3]
    deleteat!(axes, self.dim)
    push!(axes, self.dim)
    return axes
end

function backward_permutation(self::Pseudo3dSliceWise)::Array{Integer}
    axes = [1, 2]
    insert!(axes, self.dim, 3)
    return axes
end

function (self::Pseudo3dSliceWise)(x)
    (ndims(x) != 3) && throw(ArgumentError("Only 3D tensors are accepted"))
    out = permutedims(x, forward_permutation(self))
    # insert channel dimention required for a 2d Flux model
    out = reshape(out, (size(out)[begin:1:end-1]..., 1, size(out)[end]))
    out = self.model2d(out)
    # getting rid of the channel dim
    out = dropdims(out, dims=ndims(out)-1)
    out = permutedims(out, backward_permutation(self))
    return out
end