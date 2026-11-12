function testMultiThreadingSolver(; arrayType=Array, scheduler=MultiDataState)
  A = rand(ComplexF32, 3, 2)
  x = rand(ComplexF32, 2, 4)
  b = A * x

  solvers = linearSolverList()
  @testset "$(solvers[i])" for i = 1:length(solvers)
    S = createLinearSolver(solvers[i], arrayType(A), iterations=100)

    x_sequential = hcat([Array(solve!(S, arrayType(b[:, j]))) for j = 1:size(b, 2)]...)
    @test x_sequential ≈ x rtol = 0.1

    x_approx = Array(solve!(S, arrayType(b), scheduler=scheduler))
    @test x_approx ≈ x rtol = 0.1

    # Does sequential/normal reco still works after multi-threading
    x_vec = Array(solve!(S, arrayType(b[:, 1])))
    @test x_vec ≈ x[:, 1] rtol = 0.1
  end

  @testset "CGNR via Adapt" begin
    # built on the CPU and moved to adaptType with Adapt.adapt
    S = Adapt.adapt(adaptType, createLinearSolver(CGNR, A, iterations=100))
    @test Array(solve!(S, adaptType(b), scheduler=scheduler)) ≈ x rtol = 0.1
    @test Array(solve!(S, adaptType(b[:, 1]))) ≈ x[:, 1] rtol = 0.1
  end

  @testset "$solver resumed after Adapt.adapt" for solver in (CGNR, FISTA, OptISTA, POGM)
    # a partly iterated matrix solve keeps its scheduler state and progress when moved
    S = createLinearSolver(solver, A, iterations=100)
    init!(S, b; scheduler=scheduler)
    iterate(S)
    S.state.active[2] = false   # a frame the scheduler no longer iterates
    S_adapted = Adapt.adapt(adaptType, S)
    @test S_adapted.state isa scheduler
    @test S_adapted.state.active == [true, false, true, true]
    @test S_adapted.state.states[1].iteration == 1
    for _ in S end
    for _ in S_adapted end
    @test Array(solversolution(S_adapted)) ≈ solversolution(S) rtol = 1e-4
  end
end

@testset "Test MultiThreading Support" begin
  @testset "$arrayType" begin
    for scheduler in [SequentialState, MultiThreadingState]
      @testset "$scheduler" begin
        testMultiThreadingSolver(; arrayType, scheduler)
      end
    end
  end
end