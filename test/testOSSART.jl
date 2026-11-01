# OS-SART: block updates normalized by row and column sums, constraints and one further term.
function testOSSART(; arrayType = Array)
  Random.seed!(1)
  M, N = 40, 12
  A = rand(Float64, M, N)
  x = rand(Float64, N)
  b = A * x

  # more blocks converge faster per pass (measured after 500 passes: 1.5e-3 with one block, 7e-10 with five)
  @testset "consistent system, $nblocks blocks" for nblocks in (1, 2, 3, 5)
    S = createLinearSolver(OSSART, arrayType(A); blocks = RowBlockOp(arrayType(A), nblocks), iterations = 2000)
    @test Array(solve!(S, arrayType(b))) ≈ x rtol = 1e-6
  end

  @testset "one pass of SIRT by hand" begin
    S = createLinearSolver(OSSART, arrayType(A); iterations = 1, relaxation = 0.8)
    r, c = vec(sum(A; dims = 2)), vec(sum(A; dims = 1))
    @test Array(solve!(S, arrayType(b))) ≈ 0.8 .* (A' * (b ./ r)) ./ c
  end

  @testset "blocks" begin
    # contiguous and random blocks converge as well
    for mode in (:contiguous, :random)
      S = createLinearSolver(OSSART, A; blocks = RowBlockOp(A, 4; mode), iterations = 500)
      @test solve!(S, b) ≈ x rtol = 1e-3
    end
    # blocks of an operator that is no matrix, and the default single block (SIRT)
    L = LinearOperator(A)
    S1 = createLinearSolver(OSSART, L; blocks = RowBlockOp([LinearOperator(A[1:20, :]), LinearOperator(A[21:M, :])]), iterations = 3)
    S2 = createLinearSolver(OSSART, A; blocks = RowBlockOp(A, 2; mode = :contiguous), iterations = 3)
    @test solve!(S1, b) ≈ solve!(S2, b)
    @test solve!(createLinearSolver(OSSART, L; iterations = 2), b) ≈ solve!(createLinearSolver(OSSART, A; iterations = 2), b)
    @test_throws DimensionMismatch OSSART(A; blocks = RowBlockOp(A[1:20, :], 2))
  end

  @testset "positivity" begin
    xs = randn(N)
    S = createLinearSolver(OSSART, arrayType(A); blocks = RowBlockOp(arrayType(A), 4), reg = PositiveRegularization(), iterations = 50)
    @test all(>=(0), Array(solve!(S, arrayType(A * xs))))
  end

  @testset "additional term after each pass" begin
    # with an L1 term the solution shrinks compared with the plain one
    S = createLinearSolver(OSSART, arrayType(A); blocks = RowBlockOp(arrayType(A), 2), iterations = 50)
    SL1 = createLinearSolver(OSSART, arrayType(A); blocks = RowBlockOp(arrayType(A), 2), reg = L1Regularization(0.01), iterations = 50)
    @test sum(abs, Array(solve!(SL1, arrayType(b)))) < sum(abs, Array(solve!(S, arrayType(b))))
    # L2 is one such term as well: its proximal map shrinks the solution after each pass
    SL2 = createLinearSolver(OSSART, arrayType(A); blocks = RowBlockOp(arrayType(A), 2), reg = L2Regularization(0.01), iterations = 50)
    @test sum(abs2, Array(solve!(SL2, arrayType(b)))) < sum(abs2, Array(solve!(S, arrayType(b))))
    @test_throws ErrorException OSSART(A; reg = [L1Regularization(0.1), L2Regularization(0.1)])
    @test_throws ErrorException OSSART(A; reg = [L1Regularization(0.1), TVRegularization(0.1; shape = (N,))])
    @test isapplicable(OSSART, [L2Regularization(0.1)]) && isapplicable(OSSART, [L1Regularization(0.1)])
    @test !isapplicable(OSSART, [L1Regularization(0.1), L2Regularization(0.1)])
  end

  @testset "warm start" begin
    S = createLinearSolver(OSSART, arrayType(A); blocks = RowBlockOp(arrayType(A), 3), iterations = 2)
    @test Array(solve!(S, arrayType(b); x0 = arrayType(x))) ≈ x
  end

  @testset "zero rows and columns" begin
    Az = copy(A)
    Az[5, :] .= 0
    Az[:, 3] .= 0
    S = createLinearSolver(OSSART, arrayType(Az); blocks = RowBlockOp(arrayType(Az), 2), iterations = 5)
    @test all(isfinite, Array(solve!(S, arrayType(Az * x))))
  end

  @test_throws ArgumentError OSSART(A; relaxation = 2)
end

@testset "OSSART: $arrayType" begin
  testOSSART(; arrayType)
end
