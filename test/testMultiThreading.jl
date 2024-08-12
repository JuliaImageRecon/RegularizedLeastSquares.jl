function testMultiThreadingSolver(; arrayType = Array, scheduler = MultiDataState)
  A = rand(ComplexF32, 3, 2)
  x = rand(ComplexF32, 2, 4)
  b = A * x

  solvers = [CGNR] # linearSolverList()
  @testset "$(solvers[i])" for i = 1:length(solvers)
    S = createLinearSolver(solvers[i], arrayType(A), iterations = 100)
    
    x_sequential = hcat([Array(solve!(S, arrayType(b[:, j]))) for j = 1:size(b, 2)]...)
    @test x_sequential ≈ x rtol = 0.1
    
    x_approx = Array(solve!(S, arrayType(b), scheduler=scheduler))
    @test x_approx ≈ x rtol = 0.1

    # Does sequential/normal reco still works after multi-threading
    x_vec = Array(solve!(S, arrayType(b[:, 1])))
    @test x_vec ≈ x[:, 1] rtol = 0.1
  end
end

@testset "Test MultiThreading Support" begin
  for arrayType in arrayTypes
    @testset "$arrayType" begin
      for scheduler in [SequentialState, MultiThreadingState]
        @testset "$scheduler" begin
          testMultiThreadingSolver(; arrayType, scheduler)
        end
      end
    end
  end
end