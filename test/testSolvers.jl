Random.seed!(12345)

function testRealLinearSolver(; arrayType=Array, elType=Float32)
    A = rand(elType, 3, 2)
    x = rand(elType, 2)
    b = A * x

    solvers = linearSolverListReal()

    @testset for solver in solvers
        @test try
            S = createLinearSolver(solver, arrayType(A), iterations=200)
            x_approx = Array(solve!(S, arrayType(b)))
            @info "Testing solver $solver: $x ≈ $x_approx"
            @test x_approx ≈ x rtol = 0.1
            true
        catch e
            @error e
            false
        end skip = arrayType != Array && solver <: AbstractDirectSolver #
    end

    # CGNR built on the CPU and moved to adaptType with Adapt.adapt
    S = Adapt.adapt(adaptType, createLinearSolver(CGNR, A, iterations=200))
    @test Array(solve!(S, adaptType(b))) ≈ x rtol = 0.1
end

function testComplexLinearSolver(; arrayType=Array, elType=Float32)
    A = rand(elType, 3, 2) + im * rand(elType, 3, 2)
    x = rand(elType, 2) + im * rand(elType, 2)
    b = A * x

    solvers = linearSolverList()

    @testset for solver in solvers
        @test try
            S = createLinearSolver(solver, arrayType(A), iterations=100)
            x_approx = Array(solve!(S, arrayType(b)))
            @info "Testing solver $solver: $x ≈ $x_approx"
            @test x_approx ≈ x rtol = 0.1
            true
        catch e
            @error e
            false
        end skip = arrayType != Array && solver <: AbstractDirectSolver
    end

    # CGNR built on the CPU and moved to adaptType with Adapt.adapt
    S = Adapt.adapt(adaptType, createLinearSolver(CGNR, A, iterations=100))
    @test Array(solve!(S, adaptType(b))) ≈ x rtol = 0.1
end

function testComplexLinearAHASolver(; arrayType=Array, elType=Float32)
    A = rand(elType, 3, 2) + im * rand(elType, 3, 2)
    x = rand(elType, 2) + im * rand(elType, 2)
    AHA = A' * A
    b = AHA * x

    solvers = filter(s -> s ∉ [DirectSolver, PseudoInverse, Kaczmarz], linearSolverListReal())

    @testset for solver in solvers
        @test try
            S = createLinearSolver(solver, nothing; AHA=arrayType(AHA), iterations=100)
            x_approx = Array(solve!(S, arrayType(b)))
            @info "Testing solver $solver: $x ≈ $x_approx"
            @test x_approx ≈ x rtol = 0.1
            true
        catch e
            @error e
            false
        end
    end

    # CGNR built on the CPU and moved to adaptType with Adapt.adapt
    S = Adapt.adapt(adaptType, createLinearSolver(CGNR, nothing; AHA=AHA, iterations=100))
    @test Array(solve!(S, adaptType(b))) ≈ x rtol = 0.1
end

function testConvexLinearSolver(; arrayType=Array, elType=Float32)
    # fully sampled operator, image and data
    N = 256
    numPeaks = 5
    F = [1 / sqrt(N) * exp(-2im * π * j * k / N) for j = 0:N-1, k = 0:N-1]
    x = zeros(N)
    for i = 1:3
        x[rand(1:N)] = rand()
    end
    b = 1 / sqrt(N) * fft(x)

    # random undersampling
    idx = sort(unique(rand(1:N, div(N, 2))))
    b = arrayType(b[idx])
    F = arrayType(F[idx, :])

    for solver in [POGM, OptISTA, FISTA, ADMM]
        reg = L1Regularization(elType(1e-3))
        S = createLinearSolver(
            solver,
            F;
            reg=reg,
            iterations=200,
            normalizeReg=NoNormalization(),
        )
        x_approx = Array(solve!(S, b))
        @info "Testing solver $solver w/o restart: relative error = $(norm(x - x_approx) / norm(x))"
        @test x ≈ x_approx rtol = 0.1

        #additionally test the gradient restarting scheme
        if solver == POGM || solver == FISTA
            S = createLinearSolver(
                solver,
                F;
                reg=reg,
                iterations=200,
                normalizeReg=NoNormalization(),
                restart=:gradient,
            )
            x_approx = Array(solve!(S, b))
            @info "Testing solver $solver w/ gradient restart: relative error = $(norm(x - x_approx) / norm(x))"
            @test x ≈ x_approx rtol = 0.1
        end

        # test invariance to the maximum eigenvalue
        reg = L1Regularization(elType(reg.λ * length(b) / norm(b, 1)))
        scale_F = 1e3
        S = createLinearSolver(
            solver,
            F .* scale_F;
            reg=reg,
            iterations=200,
            normalizeReg=MeasurementBasedNormalization(),
        )
        x_approx = Array(solve!(S, b))
        x_approx .*= scale_F
        @info "Testing solver $solver w/o restart and after re-scaling: relative error = $(norm(x - x_approx) / norm(x))"
        @test x ≈ x_approx rtol = 0.1
    end

    # test ADMM with option vary_rho
    solver = ADMM
    reg = L1Regularization(elType(1.e-3))
    S = createLinearSolver(
        solver,
        F;
        reg=reg,
        iterations=200,
        normalizeReg=NoNormalization(),
        rho=1e6,
        vary_rho=:balance,
        verbose=false,
    )
    x_approx = Array(solve!(S, b))
    @info "Testing solver $solver: relative error = $(norm(x - x_approx) / norm(x))"
    @test x ≈ x_approx rtol = 0.1

    S = createLinearSolver(
        solver,
        F;
        reg=reg,
        iterations=200,
        normalizeReg=NoNormalization(),
        rho=1e-6,
        vary_rho=:balance,
        verbose=false,
    )
    x_approx = Array(solve!(S, b))
    @info "Testing solver $solver: relative error = $(norm(x - x_approx) / norm(x))"
    @test x ≈ x_approx rtol = 0.1

    # the PnP scheme only increases rho, hence we only test it with a small initial rho
    S = createLinearSolver(
        solver,
        F;
        reg=reg,
        iterations=200,
        normalizeReg=NoNormalization(),
        rho=1e-6,
        vary_rho=:PnP,
        verbose=false,
    )
    x_approx = Array(solve!(S, b))
    @info "Testing solver $solver: relative error = $(norm(x - x_approx) / norm(x))"
    @test x ≈ x_approx rtol = 0.1

    ##
    solver = SplitBregman
    reg = L1Regularization(elType(2e-3))
    S = createLinearSolver(
        solver,
        F;
        reg=reg,
        iterations=5,
        iterationsInner=40,
        rho=1.0,
        normalizeReg=NoNormalization(),
    )
    x_approx = Array(solve!(S, b))
    @info "Testing solver $solver: relative error = $(norm(x - x_approx) / norm(x))"
    @test x ≈ x_approx rtol = 0.1

    reg = L1Regularization(elType(reg.λ * length(b) / norm(b, 1)))
    S = createLinearSolver(
        solver,
        F;
        reg=reg,
        iterations=5,
        iterationsInner=40,
        rho=1.0,
        normalizeReg=MeasurementBasedNormalization(),
    )
    x_approx = Array(solve!(S, b))
    @info "Testing solver $solver: relative error = $(norm(x - x_approx) / norm(x))"
    @test x ≈ x_approx rtol = 0.1

    #=
    solver = PrimalDualSolver
    reg = [L1Regularization(1.e-4), TVRegularization(1.e-4, shape = (0,0))]
    FR = [real.(F ./ norm(F)); imag.(F ./ norm(F))]
    bR = [real.(b ./ norm(F)); imag.(b ./ norm(F))]
    S = createLinearSolver(
        solver,
        FR;
        reg = reg,
        iterations = 1000,
    )
    x_approx = solve!(S, bR)
    @info "Testing solver $solver: relative error = $(norm(x - x_approx) / norm(x))"
    @test x ≈ x_approx rtol = 0.1
    =#
end

function testVerboseSolvers(; arrayType=Array, elType=Float32)
    A = rand(elType, 3, 2)
    x = rand(elType, 2)
    b = A * x

    solvers = [ADMM, FISTA, POGM, OptISTA, SplitBregman]

    for solver in solvers
        @test try
            S = createLinearSolver(solver, arrayType(A), iterations=3, verbose=true)
            solve!(S, arrayType(b))
            true
        catch e
            @error e
            false
        end
    end
end

# CGNR started from x0 converges to the same (regularized) least-squares solution and leaves the
# data unchanged, with the forward operator and with only the normal operator
function testCGNRStartValue(; arrayType=Array, elType=Float64)
    A = randn(elType, 20, 8)   # well conditioned: CG takes at most 8 steps here
    b = randn(elType, 20)
    x0 = randn(elType, 8)
    for λ in (zero(elType), elType(0.5))
        x_ls = (A' * A + λ * I) \ (A' * b)
        S = createLinearSolver(CGNR, arrayType(A); reg=L2Regularization(λ), iterations=50)
        b_d = arrayType(b)
        x_approx = Array(solve!(S, b_d; x0=arrayType(x0)))
        @test x_approx ≈ x_ls rtol = sqrt(eps(elType))
        @test Array(b_d) == b

        S = createLinearSolver(CGNR, nothing; AHA=arrayType(A' * A), reg=L2Regularization(λ), iterations=50)
        x_approx = Array(solve!(S, arrayType(A' * b); x0=arrayType(x0)))
        @test x_approx ≈ x_ls rtol = sqrt(eps(elType))

        # the same solvers built on the CPU and moved to adaptType with Adapt.adapt
        S = Adapt.adapt(adaptType, createLinearSolver(CGNR, A; reg=L2Regularization(λ), iterations=50))
        @test Array(solve!(S, adaptType(b); x0=adaptType(x0))) ≈ x_ls rtol = sqrt(eps(elType))
        S = Adapt.adapt(adaptType, createLinearSolver(CGNR, nothing; AHA=A' * A, reg=L2Regularization(λ), iterations=50))
        @test Array(solve!(S, adaptType(A' * b); x0=adaptType(x0))) ≈ x_ls rtol = sqrt(eps(elType))
    end
end

# A partly iterated CGNR with a normalized, auto-scaled L2 term and a transformed projection, moved with
# Adapt.adapt, finishes with the same result as the CPU solver it was moved from
function testCGNRAdaptResume(; elType=Float64)
    A = randn(elType, 20, 8)
    b = randn(elType, 20)
    L2 = AutoScaledRegularization(L2Regularization(elType(0.1)))
    L2.factor = elType(0.5)   # a populated auto-scaling cache
    Q = Matrix(qr(randn(elType, 8, 8)).Q)
    S = createLinearSolver(CGNR, A; reg=[L2, TransformedRegularization(PositiveRegularization(), Q)],
        normalizeReg=MeasurementBasedNormalization(), iterations=6)
    init!(S, b)
    iterate(S)
    S_adapted = Adapt.adapt(adaptType, S)
    @test S_adapted.state.x isa adaptType
    @test S_adapted.constr[1].trafo isa adaptType
    @test λ(S_adapted.L2) == λ(S.L2)
    for _ in S end
    for _ in S_adapted end
    @test Array(solversolution(S_adapted)) ≈ solversolution(S) rtol = 1e-4
end

# A partly iterated ADMM with varying rho, a transformed L1 term and a transformed projection, moved with
# Adapt.adapt, finishes with the same result as the CPU solver it was moved from
function testADMMAdaptResume(; elType=Float64)
    rT = real(elType)
    A = randn(elType, 20, 8)
    Q1, Q2 = (Matrix{elType}(qr(randn(rT, 8, 8)).Q) for _ in 1:2)
    # measurements of images that are positive after Q2, so the projected solution does not collapse to zero
    B = A * Q2' * rand(rT, 8, 2)
    b = B[:, 1]
    reg = [TransformedRegularization(L1Regularization(rT(1e-3)), Q1), L1Regularization(rT(1e-3)), TransformedRegularization(PositiveRegularization(), Q2)]
    regTrafo = [randn(elType, 8, 8), Matrix{elType}(I, 8, 8)]
    kwargs = (; reg, regTrafo, rho=1e6, vary_rho=:balance, iterations=50)
    for (S, rhs) in ((createLinearSolver(ADMM, A; kwargs...), b), (createLinearSolver(ADMM, nothing; AHA=A' * A, kwargs...), A' * b))
        init!(S, rhs)
        for _ in 1:10
            S.state.ρ != S.rho && break
            iterate(S)
        end
        @test S.state.ρ != S.rho
        S_adapted = Adapt.adapt(adaptType, S)
        state, cgVars = S_adapted.state, S_adapted.state.cgStateVars
        @test S_adapted.AHA isa adaptType
        @test all(t -> t isa adaptType, S_adapted.regTrafo)
        @test S_adapted.reg[1].trafo isa adaptType
        @test S_adapted.proj[1].trafo isa adaptType
        @test all(v -> v isa adaptType, (state.x, state.xᵒˡᵈ, state.β, state.β_y, cgVars.u, cgVars.r, cgVars.c))
        @test all(v -> v isa adaptType, [state.z; state.zᵒˡᵈ; state.u; state.uᵒˡᵈ])
        @test λ(S_adapted.reg[1]) == λ(S.reg[1])
        # rho and the convergence values stay on the host as independent copies
        for f in (:ρ, :rᵏ, :sᵏ, :ɛᵖʳⁱ, :ɛᵈᵘᵃ, :Δ)
            @test getfield(state, f) isa Vector && isequal(getfield(state, f), getfield(S.state, f)) && getfield(state, f) !== getfield(S.state, f)
        end
        ρ_adapted, ρ_source = copy(S_adapted.state.ρ), copy(S.state.ρ)
        for _ in S_adapted end
        @test S_adapted.state.ρ != ρ_adapted
        @test S.state.ρ == ρ_source
        for _ in S end
        @test solversolution(S_adapted) isa adaptType
        @test Array(solversolution(S_adapted)) ≈ solversolution(S) rtol = 1e-4
    end
    # a fresh solver solves vector and matrix right-hand sides on the target
    S = createLinearSolver(ADMM, A; kwargs...)
    S_adapted = Adapt.adapt(adaptType, S)
    @test S_adapted.A isa adaptType
    @test Array(solve!(S_adapted, adaptType(b))) ≈ solve!(S, b) rtol = 1e-4
    @test Array(solve!(S_adapted, adaptType(B))) ≈ solve!(S, B) rtol = 1e-4
    # a solver holding the states of several columns is not adapted
    @test_throws Exception Adapt.adapt(adaptType, S)
end

# DirectSolver and PseudoInverse with an L2 term and a transformed projection, moved with Adapt.adapt. DirectSolver
# factorizes at each solve, so only its fields are checked; PseudoInverse solves with its stored SVD factors.
function testDirectAdapt(; elType=Float64)
    rT = real(elType)
    A = randn(elType, 20, 8)
    b = randn(elType, 20)
    Q = Matrix{elType}(qr(randn(rT, 8, 8)).Q)
    reg() = [L2Regularization(rT(0.1)), TransformedRegularization(PositiveRegularization(), Q)]
    fields(S) = (S.state.x, S.state.b, S.proj[1].trafo, S isa DirectSolver ? (S.A,) : (S.svd.U, S.svd.S, S.svd.Vt))

    S = createLinearSolver(DirectSolver, A; reg=reg(), normalizeReg=MeasurementBasedNormalization())
    source = deepcopy(fields(S))
    S_adapted = Adapt.adapt(adaptType, S)
    @test all(v -> v isa adaptType, (S_adapted.A, S_adapted.state.x, S_adapted.state.b, S_adapted.proj[1].trafo))
    @test λ(S_adapted.l2) == λ(S.l2)
    @test S_adapted.normalizeReg isa MeasurementBasedNormalization
    @test isequal(fields(S), source)

    S = createLinearSolver(PseudoInverse, A; reg=reg())
    source = deepcopy(fields(S))
    S_adapted = Adapt.adapt(adaptType, S)
    F = S_adapted.svd
    @test all(v -> v isa adaptType, (F.U, F.S, F.Vt, S_adapted.state.x, S_adapted.state.b, S_adapted.proj[1].trafo))
    @test (Array(F.U), Array(F.S), Array(F.Vt)) == (S.svd.U, S.svd.S, S.svd.Vt)
    @test λ(S_adapted.l2) == λ(S.l2)
    @test isequal(fields(S), source)
    @test Array(solve!(S_adapted, adaptType(b))) ≈ solve!(S, b) rtol = 1e-4
    @test (S.svd.U, S.svd.S, S.svd.Vt) == source[4]
end

@testset "Test Solvers" begin
    @testset "$arrayType" begin
        for elType in [Float32, Float64]
            @testset "Real Linear Solver: $elType" begin
                testRealLinearSolver(; arrayType, elType)
            end

            @testset "Complex Linear Solver: $elType" begin
                testComplexLinearSolver(; arrayType, elType)
            end

            @testset "Complex Linear Solver w/ AHA Interface: $elType" begin
                testComplexLinearAHASolver(; arrayType, elType)
            end

            @testset "General Convex Solver: $elType" begin
                testConvexLinearSolver(; arrayType, elType)
            end

            @testset "CGNR with start value: $elType" begin
                testCGNRStartValue(; arrayType, elType)
            end

            @testset "CGNR resumed after Adapt.adapt: $elType" begin
                testCGNRAdaptResume(; elType)
            end

            @testset "ADMM resumed after Adapt.adapt: $T" for T in (elType, complex(elType))
                testADMMAdaptResume(; elType=T)
            end

            @testset "DirectSolver and PseudoInverse after Adapt.adapt: $T" for T in (elType, complex(elType))
                testDirectAdapt(; elType=T)
            end
        end
    end
    testVerboseSolvers()
end