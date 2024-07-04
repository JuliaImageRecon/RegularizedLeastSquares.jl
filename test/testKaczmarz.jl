Random.seed!(12345)

@testset "Test Kaczmarz" begin
  for arrayType in arrayTypes
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
        A = rand(3, 2) + im * rand(3, 2)
        x = rand(2) + im * rand(2)
        b = A * x

        regMatrix = rand(2) # Tikhonov matrix

        solver = Kaczmarz
        S = createLinearSolver(solver, arrayType(A), iterations=200, reg=[L2Regularization(arrayType(regMatrix))])
        x_approx = Array(solve!(S, arrayType(b)))
        #@info "Testing solver $solver ...: $x  == $x_approx"
        @test norm(x - x_approx) / norm(x) ≈ 0 atol = 0.1

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

        S = createLinearSolver(solver, arrayType(A), iterations=100, reg=[L2Regularization(arrayType(regMatrix))])
        x_matrix = Array(solve!(S, arrayType(b)))

        # use standard reconstruction
        S = createLinearSolver(solver, arrayType(A * Diagonal(1 ./ sqrt.(regMatrix))), iterations=100)
        x_approx = Array(solve!(S, arrayType(b))) ./ sqrt.(regMatrix)

        # test
        #@info "Testing solver $solver ...: $x_matrix  == $x_approx"
        @test norm(x_approx - x_matrix) / norm(x_approx) ≈ 0 atol = 0.1
      end

      @testset "Kaczmarz Weighting Matrix" begin
        # TODO does not work on GPU atm, see https://github.com/JuliaGPU/GPUArrays.jl/issues/543
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
        S_weighted = createLinearSolver(solver, *(ProdOp, w, A), iterations=200, reg=reg)
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
      end
    end
  end
end
