@testset "Creation of solvers" begin
  @test_logs (:warn, Regex("The following arguments were passed but filtered out: testKwarg*")) createLinearSolver(Kaczmarz, zeros(42, 42), testKwarg=1337)
end