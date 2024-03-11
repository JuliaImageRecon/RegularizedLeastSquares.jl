
@testset "PnP Constructor" begin
    model(x) = x
    # reduced constructor, checking defaults
    pnp_reg = PnPRegularization(model, [2])
    @assert pnp_reg.λ == 1.0
    @assert pnp_reg.model == model
    @assert pnp_reg.shape == [2]
    @assert pnp_reg.input_transform == RegularizedLeastSquares.MinMaxTransform
    @assert pnp_reg.ignoreIm == false
    # full constructor
    pnp_reg = PnPRegularization(0.1; model=model, shape=[2], input_transform=x -> x, ignoreIm=true)
    # full constructor defaults
    pnp_reg = PnPRegularization(0.1; model=model, shape=[2])
    @assert pnp_reg.input_transform == RegularizedLeastSquares.MinMaxTransform
    @assert pnp_reg.ignoreIm == false
    # unnecessary kwargs are ignored
    pnp_reg = PnPRegularization(0.1; model=model, shape=[2], input_transform=x -> x, ignoreIm=true, sMtHeLsE=1)
end


@testset "PnP Compatibility" begin
    supported_solvers = [Kaczmarz, ADMM]
    A = rand(3, 2)
    x = rand(2)
    pnp_reg = PnPRegularization(x -> x, [2])
    b = A * x

    for solver in supported_solvers
        S = createLinearSolver(solver, A, iterations=2; reg=[pnp_reg])
        x_approx = solve!(S, b)
        @info "PnP Regularization and $solver Compatibility"
    end
end


@testset "PnP Prox Real" begin
    pnp_reg = PnPRegularization(0.1; model=x -> zeros(eltype(x), size(x)), shape=[2], input_transform=RegularizedLeastSquares.IdentityTransform)
    out = prox!(pnp_reg, [1.0, 2.0], 0.1)
    @info out
    @assert out == [0.9, 1.8]
end


@testset "PnP Prox Complex" begin
    # ignoreIm = false
    pnp_reg = PnPRegularization(
        0.1; model=x -> zeros(eltype(x), size(x)), shape=[2],
        input_transform=RegularizedLeastSquares.IdentityTransform
    )
    out = prox!(pnp_reg, [1.0 + 1.0im, 2.0 + 2.0im], 0.1)
    @assert real(out) == [0.9, 1.8]
    @assert imag(out) == [0.9, 1.8]
    # ignoreIm = true
    pnp_reg = PnPRegularization(
        0.1; model=x -> zeros(eltype(x), size(x)), shape=[2],
        input_transform=RegularizedLeastSquares.IdentityTransform,
        ignoreIm=true
    )
    out = prox!(pnp_reg, [1.0 + 1.0im, 2.0 + 2.0im], 0.1)
    @assert real(out) == [0.9, 1.8]
    @assert imag(out) == [1.0, 2.0]
end


@testset "PnP Prox λ clipping" begin
    pnp_reg = PnPRegularization(0.1; model=x -> zeros(eltype(x), size(x)), shape=[2], input_transform=RegularizedLeastSquares.IdentityTransform)
    out = @test_warn "$(typeof(pnp_reg)) was given λ with value 1.5. Valid range is [0, 1]. λ changed to temp" prox!(pnp_reg, [1.0, 2.0], 1.5)
    @assert out == [0.0, 0.0]
    out = @test_warn "$(typeof(pnp_reg)) was given λ with value -1.5. Valid range is [0, 1]. λ changed to temp" prox!(pnp_reg, [1.0, 2.0], -1.5)
    @assert out == [1.0, 2.0]
end