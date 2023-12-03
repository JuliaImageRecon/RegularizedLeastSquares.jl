Random.seed!(12345)

@testset "Real Linear Solver" begin
    A = rand(3, 2)
    x = rand(2)
    b = A * x

    solvers = linearSolverListReal()

    for solver in solvers
        S = createLinearSolver(
            solver,
            A,
            iterations = 200,
        )
        x_approx = solve(S, b)
        @info "Testing solver $solver: $x ≈ $x_approx"
        @test norm(x - x_approx) / norm(x) ≈ 0 atol = 0.1
    end
end

@testset "Complex Linear Solver" begin
    A = rand(3, 2) + im * rand(3, 2)
    x = rand(2) + im * rand(2)
    b = A * x

    solvers = linearSolverList()

    for solver in solvers
        S = createLinearSolver(solver, A, iterations = 100)
        x_approx = solve(S, b)
        @info "Testing solver $solver ...: $x  == $x_approx"
        @test norm(x - x_approx) / norm(x) ≈ 0 atol = 0.1
    end
end

@testset "General Convex Solver" begin
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
    b = b[idx]
    F = F[idx, :]

    for solver in [POGM, OptISTA, FISTA, ADMM]
        reg = L1Regularization(1e-3)
        S = createLinearSolver(
            solver,
            F;
            reg = reg,
            iterations = 200,
            normalizeReg = NoNormalization(),
        )
        x_approx = solve(S, b)
        @info "Testing solver $solver w/o restart: relative error = $(norm(x - x_approx) / norm(x))"
        @test x ≈ x_approx rtol = 0.1

        #additionally test the gradient restarting scheme
        if solver == POGM || solver == FISTA
            S = createLinearSolver(
                solver,
                F;
                reg = reg,
                iterations = 200,
                normalizeReg = NoNormalization(),
                restart = :gradient,
            )
            x_approx = solve(S, b)
            @info "Testing solver $solver w/ gradient restart: relative error = $(norm(x - x_approx) / norm(x))"
            @test x ≈ x_approx rtol = 0.1
        end

        # test invariance to the maximum eigenvalue
        reg = L1Regularization(reg.λ * length(b) / norm(b, 1))
        scale_F = 1e3
        S = createLinearSolver(
            solver,
            F .* scale_F;
            reg = reg,
            iterations = 200,
            normalizeReg = MeasurementBasedNormalization(),
        )
        x_approx = solve(S, b)
        x_approx .*= scale_F
        @info "Testing solver $solver w/o restart and after re-scaling: relative error = $(norm(x - x_approx) / norm(x))"
        @test x ≈ x_approx rtol = 0.1
    end

    # test ADMM with option vary_rho
    solver = ADMM
    reg = L1Regularization(1.e-3)
    S = createLinearSolver(
        solver,
        F;
        reg = reg,
        iterations = 200,
        normalizeReg = NoNormalization(),
        rho = 1e6,
        vary_rho = :balance,
        verbose = false,
    )
    x_approx = solve(S, b)
    @info "Testing solver $solver: relative error = $(norm(x - x_approx) / norm(x))"
    @test x ≈ x_approx rtol = 0.1

    S = createLinearSolver(
        solver,
        F;
        reg = reg,
        iterations = 200,
        normalizeReg = NoNormalization(),
        rho = 1e-6,
        vary_rho = :balance,
        verbose = false,
    )
    x_approx = solve(S, b)
    @info "Testing solver $solver: relative error = $(norm(x - x_approx) / norm(x))"
    @test x ≈ x_approx rtol = 0.1

    # the PnP scheme only increases rho, hence we only test it with a small initial rho
    S = createLinearSolver(
        solver,
        F;
        reg = reg,
        iterations = 200,
        normalizeReg = NoNormalization(),
        rho = 1e-6,
        vary_rho = :PnP,
        verbose = false,
    )
    x_approx = solve(S, b)
    @info "Testing solver $solver: relative error = $(norm(x - x_approx) / norm(x))"
    @test x ≈ x_approx rtol = 0.1

    ##
    solver = SplitBregman
    reg = L1Regularization(1.e-3)
    S = createLinearSolver(
        solver,
        F;
        reg = reg,
        iterations = 5,
        iterationsInner = 40,
        rho = 1.0,
        normalizeReg = NoNormalization(),
    )
    x_approx = solve(S, b)
    @info "Testing solver $solver: relative error = $(norm(x - x_approx) / norm(x))"
    @test x ≈ x_approx rtol = 0.1

    reg = L1Regularization(reg.λ * length(b) / norm(b, 1))
    S = createLinearSolver(
        solver,
        F;
        reg = reg,
        iterations = 5,
        iterationsInner = 40,
        rho = 1.0,
        normalizeReg = MeasurementBasedNormalization(),
    )
    x_approx = solve(S, b)
    @info "Testing solver $solver: relative error = $(norm(x - x_approx) / norm(x))"
    @test x ≈ x_approx rtol = 0.1

    ##
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
    x_approx = solve(S, bR)
    @info "Testing solver $solver: relative error = $(norm(x - x_approx) / norm(x))"
    @test x ≈ x_approx rtol = 0.1
end
