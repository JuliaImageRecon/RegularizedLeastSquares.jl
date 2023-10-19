using Flux


struct DenseLayer
    # public
    in_channels::Integer
    extra_channels::Integer
    bias::Bool
    ndim::Integer
    kernel_size::Integer
    # private
    conv

    function DenseLayer(;
        in_channels::Integer,
        extra_channels::Integer,
        bias::Bool=true,
        ndim::Integer=2,
        kernel_size::Integer=3,
        σ::Function=Flux.relu
    )
        conv = Flux.Conv(
            ntuple(i -> kernel_size, ndim),
            in_channels => extra_channels,
            σ;
            pad=Flux.SamePad(),
            bias=bias
        )
        new(
            in_channels,
            extra_channels,
            bias,
            ndim,
            kernel_size,
            conv,
        )
    end
end
@Flux.functor DenseLayer

function (model::DenseLayer)(x)
    out = model.conv(x)
    return cat(x, out, dims=model.ndim+1)
end


struct ResidualDenseBlock
    # public
    in_channels::Integer
    n_layers::Integer
    growth_rate::Integer
    bias::Bool
    ndim::Integer
    kernel_size::Integer
    # private
    local_res_block
    out_conv

    function ResidualDenseBlock(;
        in_channels::Integer,
        n_layers::Integer,
        growth_rate::Integer,
        bias::Bool=true,
        ndim::Integer=2,
        kernel_size::Integer=3,
        σ::Function=Flux.relu
    )
        dense_layers = Flux.Chain([
            DenseLayer(
                in_channels=in_channels+(i-1)*growth_rate,
                extra_channels=growth_rate,
                bias=bias,
                ndim=ndim,
                kernel_size=kernel_size,
                σ=σ
            )
            for i in 1:n_layers
        ])

        out_conv = Flux.Conv(
            ntuple(i -> 1, ndim),
            (in_channels + n_layers*growth_rate) => in_channels,
            identity;
            pad=Flux.SamePad(),
            bias=bias
        )
        
        new(
            in_channels,
            n_layers,
            growth_rate,
            bias,
            ndim,
            kernel_size,
            dense_layers,
            out_conv
        )
    end 
end
@Flux.functor ResidualDenseBlock

function (model::ResidualDenseBlock)(x)
    out = model.local_res_block(x)
    return x + model.out_conv(out)
end



mutable struct rdnDenoiserResRelu
    # public
    in_channels::Integer
    out_channels::Integer
    n_features::Integer
    n_blocks::Integer
    n_layers_in_block::Integer
    growth_rate::Integer
    bias::Bool
    ndim::Integer
    kernel_size::Integer
    out_transform::Function
    # private
    conv0
    conv1
    conv2
    conv3
    conv4
    rdbs
    function rdnDenoiserResRelu(;
        in_channels::Integer,
        out_channels::Integer,
        n_features::Integer,
        n_blocks::Integer,
        n_layers_in_block::Integer,
        growth_rate::Integer,
        bias::Bool=true,
        σ=Flux.relu,
        ndim::Integer=2,
        kernel_size::Integer=3,
        out_transform::Function=Function=Flux.relu,
    )
        conv0 = Flux.Conv(
            ntuple(i -> kernel_size, ndim),
            in_channels => n_features,
            Flux.identity
            ;
            pad=Flux.SamePad(),
            bias=bias
        )
        conv1 = Flux.Conv(
            ntuple(i -> kernel_size, ndim),
            n_features => n_features,
            Flux.identity
            ;
            pad=Flux.SamePad(),
            bias=bias
        )
        rdbs = [
            ResidualDenseBlock(
                in_channels=n_features,
                n_layers=n_layers_in_block,
                growth_rate=growth_rate,
                bias=bias,
                ndim=ndim,
                kernel_size=kernel_size,
                σ=σ,
            )
            for _ in 1:n_blocks
        ]
        conv2 = Flux.Conv(
            ntuple(i -> 1, ndim),
            n_blocks*n_features => n_features,
            Flux.identity
            ;
            pad=Flux.SamePad(),
            bias=bias
        )
        conv3 = Flux.Conv(
            ntuple(i -> kernel_size, ndim),
            n_features => n_features,
            Flux.identity
            ;
            pad=Flux.SamePad(),
            bias=bias
        )
        conv4 = Flux.Conv(
            ntuple(i -> kernel_size, ndim),
            n_features => out_channels,
            Flux.identity
            ;
            pad=Flux.SamePad(),
            bias=bias
        )        

        new(
            # public args
            in_channels,
            out_channels,
            n_features,
            n_blocks,
            n_layers_in_block,
            growth_rate,
            bias,
            ndim,
            kernel_size,
            out_transform,
            # private args
            conv0,
            conv1,
            conv2,
            conv3,
            conv4,
            rdbs
        )
    end
end
@Flux.functor rdnDenoiserResRelu

rdnDenoiserResRelu() = rdnDenoiserResRelu(
    in_channels=1,
    out_channels=1,
    n_features=12,
    n_blocks=4,
    n_layers_in_block=4,
    growth_rate=12,
    bias=true
)

function (model::rdnDenoiserResRelu)(x)
    x0 = x
    x = model.conv0(x)
    residual0 = x
    x = model.conv1(x)
    rdb_outs = []
    for layer in model.rdbs
        x = layer(x)
        push!(rdb_outs, x)
    end
    x = cat(rdb_outs..., dims=model.ndim+1)
    x = model.conv2(x)
    x = model.conv3(x) + residual0
    return model.out_transform(model.conv4(x) + x0)
end
