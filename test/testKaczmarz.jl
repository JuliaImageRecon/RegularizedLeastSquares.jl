Random.seed!(12345)

@testset "Test Kaczmarz" begin
  @testset "$arrayType" begin
    for T in [Float32, Float64, ComplexF32, ComplexF64]
      @testset "test Kaczmarz update $T" begin
        # set up
        M = 127
        N = 16

        A = arrayType(rand(T, M, N))
        Aᵀ = transpose(A)
        b = arrayType(zeros(T, M))
        β = rand(T)
        k = rand(1:N)
        # end set up

        RegularizedLeastSquares.kaczmarz_update!(Aᵀ, b, k, β)
        @test Array(b) ≈ β * conj(Array(A[:, k]))

        # set up
        M = 127
        N = 16

        A = arrayType(rand(T, N, M))
        b = arrayType(zeros(T, M))
        β = rand(T)
        k = rand(1:N)
        # end set up

        RegularizedLeastSquares.kaczmarz_update!(A, b, k, β)
        @test Array(b) ≈ β * conj(Array(A[k, :]))
      end
    end

    # Test Tikhonov regularization matrix
    @testset "Kaczmarz Tikhonov matrix" begin
      ## Test spatial regularization
      M = 12
      N = 8
      A = rand(M, N) + im * rand(M, N)
      x = rand(N) + im * rand(N)
      b = A * x

      # regularization
      λ = rand(1)
      regMatrix = rand(N)

      # @show A, x, regMatrix
      # use regularization matrix
      S = createLinearSolver(Kaczmarz, arrayType(A), iterations=100, reg=[L2Regularization(arrayType(regMatrix))])
      x_matrix = Array(solve!(S, arrayType(b)))

      # use standard reconstruction
      S = createLinearSolver(Kaczmarz, arrayType(A * Diagonal(1 ./ sqrt.(regMatrix))), reg=[L2Regularization(1.0)], iterations=100)
      x_approx = Array(solve!(S, arrayType(b))) ./ sqrt.(regMatrix)

      # test
      #@info "Testing solver $solver ...: $x_matrix  == $x_approx"
      @test norm(x_approx - x_matrix) / norm(x_approx) ≈ 0 atol = 0.1

      # Compare reg. matrix of equal elements to standard reco
      λ = rand()
      S = createLinearSolver(Kaczmarz, arrayType(A), iterations=100, reg=[L2Regularization(λ)])
      x_standard = Array(solve!(S, arrayType(b)))

      S = createLinearSolver(Kaczmarz, arrayType(A), iterations=100, reg=[L2Regularization(arrayType(fill(λ, N)))])
      x_matrix = Array(solve!(S, arrayType(b)))
      @test isapprox(x_standard, x_matrix)
    end

    @testset "Kaczmarz Weighting Matrix" begin
      M = 12
      N = 8
      A = rand(M, N) + im * rand(M, N)
      x = rand(N) + im * rand(N)
      b = A * x
      w = WeightingOp(rand(M))
      d = diagm(w.weights)

      reg = L2Regularization(rand())

      solver = Kaczmarz
      S = createLinearSolver(solver, d * A, iterations=200, reg=reg)
      S_weighted = createLinearSolver(solver, w ∘ A, iterations=200, reg=reg)
      x_approx = solve!(S, d * b)
      x_weighted = solve!(S_weighted, d * b)
      #@info "Testing solver $solver ...: $x  == $x_approx"
      @test isapprox(x_approx, x_weighted)
    end


    # Test Kaczmarz parameters
    @testset "Kaczmarz parameters" begin
      M = 12
      N = 8
      A = rand(M, N) + im * rand(M, N)
      x = rand(N) + im * rand(N)
      b = A * x

      solver = Kaczmarz
      S = createLinearSolver(solver, arrayType(A), iterations=200)
      x_approx = Array(solve!(S, arrayType(b)))
      @test norm(x - x_approx) / norm(x) ≈ 0 atol = 0.1

      S = createLinearSolver(solver, arrayType(A), iterations=200, shuffleRows=true)
      x_approx = Array(solve!(S, arrayType(b)))
      @test norm(x - x_approx) / norm(x) ≈ 0 atol = 0.1

      S = createLinearSolver(solver, arrayType(A), iterations=2000, randomized=true)
      x_approx = Array(solve!(S, arrayType(b)))
      @test norm(x - x_approx) / norm(x) ≈ 0 atol = 0.1

      if arrayType == Array # Greedy Kaczmarz is not defined for GPUs atm
        S = createLinearSolver(solver, arrayType(A), iterations=200, greedy_randomized=true)
        x_approx = Array(solve!(S, arrayType(b)))
        @test norm(x - x_approx) / norm(x) ≈ 0 atol = 0.1
      end

      # Normalization
      for strategy in [SystemMatrixBasedNormalization(), MeasurementBasedNormalization()]
        l2 = L2Regularization(0.1)
        S = createLinearSolver(solver, arrayType(A), iterations=200, randomized=true, reg = l2, normalizeReg = strategy)
        x_approx = Array(solve!(S, arrayType(b)))
        @test norm(x - x_approx) / norm(x) ≈ 0 atol = 0.3
      end
    end

    # Solvers built on the CPU and moved to adaptType with Adapt.adapt, fresh and after one iteration
    arrayType == Array && @testset "Kaczmarz after Adapt.adapt" begin
      M = 12
      N = 8
      A = rand(M, N) + im * rand(M, N)
      x = rand(N) + im * rand(N)
      b = A * x
      # device arrays move and keep their values, host vectors are equal copies and all other fields are unchanged
      function testAdaptedState(state, source, device, host)
        for f in fieldnames(typeof(state))
          v, v₀ = getfield(state, f), getfield(source, f)
          if f in device
            @test v isa adaptType && Array(v) == v₀
          elseif f in host
            @test v isa Vector && v == v₀ && v !== v₀
          else
            @test isequal(v, v₀)
          end
        end
      end
      for kw in ((;), (; shuffleRows=true), (; randomized=true), (; reg=L2Regularization(rand(N))), (; reg=[L2Regularization(0.1), L1Regularization(1e-3)]),
                 (; randomized=true, reg=L2Regularization(0.1), normalizeReg=MeasurementBasedNormalization()))
        S = Adapt.adapt(adaptType, createLinearSolver(Kaczmarz, A; iterations=200, kw...))
        @test Array(solve!(S, adaptType(b))) ≈ solve!(createLinearSolver(Kaczmarz, A; iterations=200, kw...), b) rtol = 1e-4

        S = createLinearSolver(Kaczmarz, A; iterations=200, kw...)
        init!(S, b)
        iterate(S)
        rng = copy(Random.default_rng())
        S_adapted = Adapt.adapt(adaptType, S)
        @test copy(Random.default_rng()) == rng
        @test parent(S_adapted.A) isa adaptType
        @test typeof(S_adapted.reg) == typeof(S.reg) && isequal(S_adapted.reg, S.reg)
        @test Adapt.adapt(Array, λ(S_adapted.L2)) == λ(S.L2)
        λ(S.L2) isa AbstractVector && @test λ(S_adapted.L2) isa adaptType
        for f in (:denom, :rowindex, :rowIndexCycle, :probabilities)
          @test getfield(S_adapted, f) isa Vector && getfield(S_adapted, f) == getfield(S, f) && getfield(S_adapted, f) !== getfield(S, f)
        end
        @test all(f -> isequal(getfield(S_adapted, f), getfield(S, f)), (:randomized, :subMatrixSize, :shuffleRows, :seed, :normalizeReg, :iterations))
        testAdaptedState(S_adapted.state, S.state, (:u, :x, :vl), (:usedIndices,))
        @test S_adapted.state.iteration == 1
        # both continue from the same external random state without being reinitialized
        Random.seed!(42); for _ in S end
        Random.seed!(42); for _ in S_adapted end
        @test Array(solversolution(S_adapted)) ≈ solversolution(S) rtol = 1e-4
      end

      # Greedy Kaczmarz only runs on the CPU: its state moves with Adapt.adapt and continues after moving back
      S = createLinearSolver(Kaczmarz, A; iterations=200, greedy_randomized=true)
      init!(S, b)
      iterate(S)
      S_adapted = Adapt.adapt(adaptType, S)
      @test S_adapted.state isa RegularizedLeastSquares.GreedyKaczmarzState
      testAdaptedState(S_adapted.state, S.state, (:u, :x, :vl, :norms, :r, :B, :diff_vec_sq, :r_probs), (:usedIndices, :U_k))
      @test S_adapted.state.iteration == 1
      S_cpu = Adapt.adapt(Array, S_adapted)
      Random.seed!(42); for _ in S end
      Random.seed!(42); for _ in S_cpu end
      @test solversolution(S_cpu) ≈ solversolution(S)
    end
  end
end
