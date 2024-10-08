Random.seed!(12345)

function testRealLinearSolver(; arrayType = Array, elType = Float32)
    A = rand(elType, 3, 2)
    x = rand(elType, 2)
    b = A * x

    solvers = linearSolverListReal()

    @testset for solver in solvers
        @test try
            S = createLinearSolver(solver, arrayType(A), iterations = 200)
            x_approx = Array(solve!(S, arrayType(b)))
            @info "Testing solver $solver: $x ≈ $x_approx"
            @test x_approx ≈ x rtol = 0.1
            true
        catch e
            @error e
            false
        end skip = arrayType != Array && solver <: AbstractDirectSolver # 
    end
end

function testComplexLinearSolver(; arrayType = Array, elType = Float32)
    A = rand(elType, 3, 2) + im * rand(elType, 3, 2)
    x = rand(elType, 2) + im * rand(elType, 2)
    b = A * x

    solvers = linearSolverList()

    @testset for solver in solvers
        @test try
            S = createLinearSolver(solver, arrayType(A), iterations = 100)
            x_approx = Array(solve!(S, arrayType(b)))
            @info "Testing solver $solver: $x ≈ $x_approx"
            @test x_approx ≈ x rtol = 0.1
            true
        catch e
            @error e
            false
        end skip = arrayType != Array && solver <: AbstractDirectSolver
    end
end

function testComplexLinearAHASolver(; arrayType = Array, elType = Float32)
    A = rand(elType, 3, 2) + im * rand(elType, 3, 2)
    x = rand(elType, 2) + im * rand(elType, 2)
    AHA = A'*A
    b = AHA * x

    solvers = filter(s -> s ∉ [DirectSolver, PseudoInverse, Kaczmarz], linearSolverListReal())

    @testset for solver in solvers
        @test try
            S = createLinearSolver(solver, nothing; AHA=arrayType(AHA), iterations = 100)
            x_approx = Array(solve!(S, arrayType(b)))
            @info "Testing solver $solver: $x ≈ $x_approx"
            @test x_approx ≈ x rtol = 0.1
            true
        catch e
            @error e
            false
        end
    end
end

function testConvexLinearSolver(; arrayType = Array, elType = Float32)
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
            reg = reg,
            iterations = 200,
            normalizeReg = NoNormalization(),
        )
        x_approx = Array(solve!(S, b))
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
            reg = reg,
            iterations = 200,
            normalizeReg = MeasurementBasedNormalization(),
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
        reg = reg,
        iterations = 200,
        normalizeReg = NoNormalization(),
        rho = 1e6,
        vary_rho = :balance,
        verbose = false,
    )
    x_approx = Array(solve!(S, b))
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
    x_approx = Array(solve!(S, b))
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
    x_approx = Array(solve!(S, b))
    @info "Testing solver $solver: relative error = $(norm(x - x_approx) / norm(x))"
    @test x ≈ x_approx rtol = 0.1

    ##
    solver = SplitBregman
    reg = L1Regularization(elType(2e-3))
    S = createLinearSolver(
        solver,
        F;
        reg = reg,
        iterations = 5,
        iterationsInner = 40,
        rho = 1.0,
        normalizeReg = NoNormalization(),
    )
    x_approx = Array(solve!(S, b))
    @info "Testing solver $solver: relative error = $(norm(x - x_approx) / norm(x))"
    @test x ≈ x_approx rtol = 0.1

    reg = L1Regularization(elType(reg.λ * length(b) / norm(b, 1)))
    S = createLinearSolver(
        solver,
        F;
        reg = reg,
        iterations = 5,
        iterationsInner = 40,
        rho = 1.0,
        normalizeReg = MeasurementBasedNormalization(),
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

function testVerboseSolvers(; arrayType = Array, elType = Float32)
    A = rand(elType, 3, 2)
    x = rand(elType, 2)
    b = A * x

    solvers = [ADMM, FISTA, POGM, OptISTA, SplitBregman]

    for solver in solvers
        @test try
            S = createLinearSolver(solver, arrayType(A), iterations = 3, verbose = true)
            solve!(S, arrayType(b))
            true
        catch e
            @error e
            false
        end
    end
end


@testset "Test Solvers" begin
    for arrayType in arrayTypes
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
            end
        end
    end
    testVerboseSolvers()
end