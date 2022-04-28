Random.seed!(12345)

@testset "Real Linear Solver" begin
    A = rand(3, 2)
    x = rand(2)
    b = A * x

    solvers = linearSolverListReal()

    for solver in solvers
        solverInfo = SolverInfo(Float64)
        S = createLinearSolver(
            solver,
            A,
            iterations = 100,
            solverInfo = solverInfo,
            shape = (2, 1),
        )
        x_approx = solve(S, b)
        @info "Testing solver $solver ...: $x  == $x_approx"
        @test norm(x - x_approx) / norm(x) ≈ 0 atol = 0.1
    end
end

@testset "Complex Linear Solver" begin
    A = rand(3, 2) + im * rand(3, 2)
    x = rand(2) + im * rand(2)
    b = A * x

    solvers = linearSolverList()

    for solver in solvers
        solverInfo = SolverInfo(ComplexF64)
        S = createLinearSolver(solver, A, iterations = 100, solverInfo = solverInfo)
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

    for solver in ["fista", "admm"]
        reg = Regularization("L1", 1e-3)
        solverInfo = SolverInfo(ComplexF64)
        S = createLinearSolver(
            solver,
            F;
            reg = reg,
            iterations = 200,
            solverInfo = solverInfo,
            normalizeReg = false,
        )
        x_approx = solve(S, b)
        @info "Testing solver $solver ...: relative error = $(norm(x - x_approx) / norm(x))"
        @test x ≈ x_approx rtol = 0.1

        reg.λ *= length(b) / norm(b, 1)
        scale_F = 1e3 # test invariance to the maximum eigenvalue
        S = createLinearSolver(
            solver,
            F .* scale_F;
            reg = reg,
            iterations = 200,
            solverInfo = solverInfo,
            normalizeReg = true,
        )
        x_approx = solve(S, b)
        x_approx .*= scale_F
        @info "Testing solver $solver ...: relative error = $(norm(x - x_approx) / norm(x))"
        @test x ≈ x_approx rtol = 0.1
    end

    # test ADMM with option vary_ρ
    solver = "admm"
    reg = Regularization("L1", 1.e-3)
    solverInfo = SolverInfo(ComplexF64)
    S = createLinearSolver(
        solver,
        F;
        reg = reg,
        iterations = 200,
        solverInfo = solverInfo,
        normalizeReg = false,
        ρ = 1e6,
        vary_ρ = :balance,
        verbose = false,
    )
    x_approx = solve(S, b)
    @info "Testing solver $solver ...: relative error = $(norm(x - x_approx) / norm(x))"
    @test x ≈ x_approx rtol = 0.1

    S = createLinearSolver(
        solver,
        F;
        reg = reg,
        iterations = 200,
        solverInfo = solverInfo,
        normalizeReg = false,
        ρ = 1e-6,
        vary_ρ = :balance,
        verbose = false,
    )
    x_approx = solve(S, b)
    @info "Testing solver $solver ...: relative error = $(norm(x - x_approx) / norm(x))"
    @test x ≈ x_approx rtol = 0.1

    # the PnP scheme only increases ρ, hence we only test it with a small initial ρ
    S = createLinearSolver(
        solver,
        F;
        reg = reg,
        iterations = 200,
        solverInfo = solverInfo,
        normalizeReg = false,
        ρ = 1e-6,
        vary_ρ = :PnP,
        verbose = false,
    )
    x_approx = solve(S, b)
    @info "Testing solver $solver ...: relative error = $(norm(x - x_approx) / norm(x))"
    @test x ≈ x_approx rtol = 0.1

    ##
    solver = "splitBregman"
    reg = Regularization("L1", 1.e-3)
    solverInfo = SolverInfo(ComplexF64)
    S = createLinearSolver(
        solver,
        F;
        reg = reg,
        iterations = 5,
        iterationsInner = 40,
        ρ = 1.0,
        solverInfo = solverInfo,
        normalizeReg = false,
    )
    x_approx = solve(S, b)
    @info "Testing solver $solver ...: relative error = $(norm(x - x_approx) / norm(x))"
    @test x ≈ x_approx rtol = 0.1

    reg.λ *= length(b) / norm(b, 1)
    S = createLinearSolver(
        solver,
        F;
        reg = reg,
        iterations = 5,
        iterationsInner = 40,
        ρ = 1.0,
        solverInfo = solverInfo,
        normalizeReg = true,
    )
    x_approx = solve(S, b)
    @info "Testing solver $solver ...: relative error = $(norm(x - x_approx) / norm(x))"
    @test x ≈ x_approx rtol = 0.1

    ##
    solver = "primaldualsolver"
    reg = [Regularization("L1", 1.e-4), Regularization("TV", 1.e-4)]
    solverInfo = SolverInfo(Float64)
    FR = [real.(F ./ norm(F)); imag.(F ./ norm(F))]
    bR = [real.(b ./ norm(F)); imag.(b ./ norm(F))]
    S = createLinearSolver(
        solver,
        FR;
        reg = reg,
        regName = ["L1", "TV"],
        iterations = 1000,
        solverInfo = solverInfo,
    )
    x_approx = solve(S, bR)
    @info "Testing solver $solver ...: relative error = $(norm(x - x_approx) / norm(x))"
    @test x ≈ x_approx rtol = 0.1
end
