@testset "Test Callbacks" begin
  A = rand(32, 32)
  x = rand(32)
  b = A * x

  iterations = 10
  solver = createLinearSolver(CGNR, A; iterations = iterations, relTol = 0.0)

  @testset "Store Solution Callback" begin
    cbk = StoreSolutionCallback()
    x_approx = solve!(solver, b; callbacks = cbk)

    @test length(cbk.solutions) == iterations + 1
    @test cbk.solutions[end] == x_approx
  end

  @testset "Compare Solution Callback" begin
    cbk = CompareSolutionCallback(x)
    x_approx = solve!(solver, b; callbacks = cbk)

    @test length(cbk.results) == iterations + 1
    @test cbk.results[1] > cbk.results[end]
  end

  @testset "Store Solution Callback" begin
    cbk = StoreConvergenceCallback()
    x_approx = solve!(solver, b; callbacks = cbk)

    @test length(first(values(cbk.convMeas))) == iterations + 1
    conv = solverconvergence(solver)
    @test cbk.convMeas[keys(conv)[1]][end] == conv[1]
  end

  @testset "Do-Syntax Callback" begin
    counter = 0

    solve!(solver, b) do solver, it
      counter +=1
    end

    @test counter == iterations + 1
  end

  @testset "Multiple Callbacks" begin
    callbacks = [StoreSolutionCallback(), StoreConvergenceCallback()]

    x_approx = solve!(solver, b; callbacks)

    cbk = callbacks[1]
    @test length(cbk.solutions) == iterations + 1
    @test cbk.solutions[end] == x_approx

    cbk = callbacks[2]
    @test length(first(values(cbk.convMeas))) == iterations + 1
    conv = solverconvergence(solver)
    @test cbk.convMeas[keys(conv)[1]][end] == conv[1]
  end

  @testset "Direct Solver Callbacks: $solverType" for solverType in [DirectSolver, PseudoInverse]
    # Direct solvers have no iterations, callbacks are expected before (0) and after (1) the solution is computed
    direct = createLinearSolver(solverType, A)

    its = Int[]
    cbk = StoreSolutionCallback()
    x_approx = solve!(direct, b; callbacks = [cbk, (_, it) -> push!(its, it)])

    @test its == [0, 1]
    @test length(cbk.solutions) == 2
    @test cbk.solutions[end] == x_approx
    @test x_approx ≈ x rtol = 1e-4

    # Solving again behaves the same
    empty!(its)
    x_again = solve!(direct, b; callbacks = (_, it) -> push!(its, it))
    @test its == [0, 1]
    @test x_again ≈ x_approx

    # Multiple measurements
    B = hcat(b, 2 .* b)
    empty!(its)
    X = solve!(direct, B; callbacks = (_, it) -> push!(its, it))
    @test its == [0, 1]
    @test X ≈ hcat(x_approx, 2 .* x_approx) rtol = 1e-4
  end

  @testset "Multiple Measurements Callbacks: $scheduler" for scheduler in [SequentialState, MultiThreadingState]
    # The schedulers must not report an additional iteration in which no state was iterated
    B = hcat(b, 2 .* b)

    its = Int[]
    solve!(solver, B; scheduler, callbacks = (_, it) -> push!(its, it))
    @test its == collect(0:iterations)

    @testset "$solverType" for solverType in [DirectSolver, PseudoInverse]
      direct = createLinearSolver(solverType, A)
      empty!(its)
      X = solve!(direct, B; scheduler, callbacks = (_, it) -> push!(its, it))
      @test its == [0, 1]
      @test X ≈ hcat(x, 2 .* x) rtol = 1e-4
    end
  end
end
