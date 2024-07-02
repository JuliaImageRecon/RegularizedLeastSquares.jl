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
end