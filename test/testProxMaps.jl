# check Thikonov proximal map
function testL2Prox(N=256; numPeaks=5, λ=0.01, arrayType=Array)
  @info "test L2-regularization"
  Random.seed!(1234)
  x = zeros(N)
  for i = 1:numPeaks
    x[rand(1:N)] = rand()
  end

  # x_l2 = 1. / (1. + 2. *λ)*x
  x_l2 = copy(x)
  x_l2 = Array(prox!(L2Regularization, arrayType(x_l2), λ))
  @test norm(x_l2 - 1.0 / (1.0 + 2.0 * λ) * x) / norm(1.0 / (1.0 + 2.0 * λ) * x) ≈ 0 atol = 0.001
  # check decrease of objective function
  @test 0.5 * norm(x - x_l2)^2 + norm(L2Regularization, x_l2, λ) <= norm(L2Regularization, x, λ)
end

# denoise a signal consisting of a number of delta peaks
function testL1Prox(N=256; numPeaks=5, σ=0.03, arrayType=Array)
  @info "test L1-regularization"
  Random.seed!(1234)
  x = zeros(N)
  for i = 1:numPeaks
    x[rand(1:N)] = (1 - 2 * σ) * rand() + 2 * σ
  end

  σ = sum(abs.(x)) / length(x) * σ
  xNoisy = x .+ σ / sqrt(2.0) * (randn(N) + 1im * randn(N))

  x_l1 = copy(xNoisy)
  x_l1 = Array(prox!(L1Regularization, arrayType(x_l1), 2 * σ))

  # solution should be better then without denoising
  @info "rel. L1 error : $(norm(x - x_l1)/ norm(x)) vs $(norm(x - xNoisy)/ norm(x))"
  @test norm(x - x_l1) <= norm(x - xNoisy)
  @test norm(x - x_l1) / norm(x) ≈ 0 atol = 0.1
  # check decrease of objective function
  @test 0.5 * norm(xNoisy - x_l1)^2 + norm(L1Regularization, x_l1, 2 * σ) <= norm(L1Regularization, xNoisy, 2 * σ)
end

# denoise a signal consisting  of multiple slices with delta peaks at the same locations
# only the last slices are noisy.
# Thus, the first slices serve as a reference to enhance denoising
function testL21Prox(N=256; numPeaks=5, numSlices=8, noisySlices=2, σ=0.05, arrayType=Array)
  @info "test L21-regularization"
  Random.seed!(1234)
  x = zeros(ComplexF64, N, numSlices)
  for i = 1:numPeaks
    x[rand(1:N), :] = (1 - 2 * σ) * rand(numSlices) .+ 2 * σ
  end
  x = vec(x)

  xNoisy = copy(x)
  noise = randn(N * noisySlices)
  σ = sum(abs.(x)) / length(x) * σ
  xNoisy[(numSlices-noisySlices)*N+1:end] .+= σ / sqrt(2.0) * (randn(N * noisySlices) + 1im * randn(N * noisySlices)) #noise

  x_l1 = copy(xNoisy)
  prox!(L1Regularization, x_l1, 2 * σ)

  x_l21 = copy(xNoisy)
  x_l21 = Array(prox!(L21Regularization, arrayType(x_l21), 2 * σ, slices=numSlices))

  # solution should be better then without denoising and with l1-denoising
  @info "rel. L21 error : $(norm(x - x_l21)/ norm(x)) vs $(norm(x - xNoisy)/ norm(x))"
  @test norm(x - x_l21) <= norm(x - xNoisy)
  @test norm(x - x_l21) <= norm(x - x_l1)
  @test norm(x - x_l21) / norm(x) ≈ 0 atol = 0.05
  # check decrease of objective function
  @test 0.5 * norm(xNoisy - x_l21)^2 + norm(L21Regularization, x_l21, 2 * σ, slices=numSlices) <= norm(L21Regularization, xNoisy, 2 * σ, slices=numSlices)
  @test 0.5 * norm(xNoisy - x_l21)^2 + norm(L21Regularization, x_l21, 2 * σ, slices=numSlices) <= 0.5 * norm(xNoisy - x_l1)^2 + norm(L21Regularization, x_l1, 2 * σ, slices=numSlices)
end

# denoise a piece-wise constant signal using TV regularization
function testTVprox(N=256; numEdges=5, σ=0.05, arrayType=Array)
  @info "test TV-regularization"
  Random.seed!(1234)
  x = zeros(ComplexF64, N, N)
  for i = 1:numEdges
    idx1 = rand(0:N-1)
    idx2 = rand(0:N-1)
    x[idx1+1:end, idx2+1:end] .+= randn()
  end
  x = vec(x)

  xNoisy = copy(x)
  σ = sum(abs.(x)) / length(x) * σ
  xNoisy[:] += σ / sqrt(2.0) * (randn(N * N) + 1im * randn(N * N))

  x_l1 = copy(xNoisy)
  prox!(L1Regularization, x_l1, 2 * σ)

  x_tv = copy(xNoisy)
  x_tv = Array(prox!(TVRegularization, arrayType(x_tv), 2 * σ, shape=(N, N), dims=1:2))

  @info "rel. TV error : $(norm(x - x_tv)/ norm(x)) vs $(norm(x - xNoisy)/ norm(x))"
  @test norm(x - x_tv) <= norm(x - xNoisy)
  @test norm(x - x_tv) <= norm(x - x_l1)
  @test norm(x - x_tv) / norm(x) ≈ 0 atol = 0.05
  # check decrease of objective function
  @test 0.5 * norm(xNoisy - x_tv)^2 + norm(TVRegularization, x_tv, 2 * σ, shape=(N, N)) <= norm(TVRegularization, xNoisy, 2 * σ, shape=(N, N))
  @test 0.5 * norm(xNoisy - x_tv)^2 + norm(TVRegularization, x_tv, 2 * σ, shape=(N, N)) <= 0.5 * norm(xNoisy - x_l1)^2 + norm(TVRegularization, x_l1, 2 * σ, shape=(N, N))
end

# denoise a signal that is piecewise constant along a given direction
function testDirectionalTVprox(N=256; numEdges=5, σ=0.05, T=ComplexF64, arrayType=Array)
  x = zeros(T, N, N)
  for i = 1:numEdges
    idx = rand(0:N-1)
    x[:, idx+1:end, :] .+= randn(T)
  end

  xNoisy = copy(x)
  σ = sum(abs.(x)) / length(x) * σ
  xNoisy .+= (σ / sqrt(2)) .* randn(T, N, N)

  x_tv = copy(xNoisy)
  x_tv = Array(reshape(prox!(TVRegularization, arrayType(vec(x_tv)), 2 * σ, shape=(N, N), dims=1), N, N))

  x_tv2 = copy(xNoisy)
  for i = 1:N
    x_tmp = x_tv2[:, i]
    prox!(TVRegularization, x_tmp, 2 * σ, shape=(N,), dims=1)
    x_tv2[:, i] .= x_tmp
  end

  # directional TV and 1d TV should yield the same result
  @test norm(x_tv - x_tv2) / norm(x) ≈ 0 atol = 1e-8
  # check decrease of error
  @test norm(x - x_tv) <= norm(x - xNoisy)

  ## cf. Condat and gradient based algorithm
  x_tv3 = copy(xNoisy)
  x_tv3 = Array(reshape(prox!(TVRegularization, vec(x_tv3), 2 * σ, shape=(N, N), dims=(1,)), N, N))
  @test norm(x_tv - x_tv3) / norm(x) ≈ 0 atol = 1e-2
end

# test enforcement of positivity constraint
function testPositive(N=256; arrayType=Array)
  @info "test positivity-constraint"
  Random.seed!(1234)
  x = randn(N) .+ 1im * randn(N)
  xPos = real.(x)
  xPos[findall(x -> x < 0, xPos)] .= 0
  xProj = copy(x)
  xProj = Array(prox!(PositiveRegularization, arrayType(xProj)))

  @test norm(xProj - xPos) / norm(xPos) ≈ 0 atol = 1.e-4
  # check decrease of objective function
  @test 0.5 * norm(x - xProj)^2 + norm(PositiveRegularization, xProj) <= norm(PositiveRegularization, x)
end

# test enforcement of "realness"-constraint
function testProj(N=1012; arrayType=Array)
  @info "test realness-constraint"
  Random.seed!(1234)
  x = randn(N) .+ 1im * randn(N)
  xReal = real.(x)
  xProj = copy(x)
  xProj = Array(prox!(ProjectionRegularization, arrayType(xProj), projFunc=x -> real(x)))
  @test norm(xProj - xReal) / norm(xReal) ≈ 0 atol = 1.e-4
  # check decrease of objective function
  @test 0.5 * norm(x - xProj)^2 + norm(ProjectionRegularization, xProj, projFunc=x -> real(x)) <= norm(ProjectionRegularization, x, projFunc=x -> real(x))
end

# test denoising of a low-rank matrix
function testNuclear(N=32, rank=2; σ=0.05, arrayType=Array)
  @info "test nuclear norm regularization"
  Random.seed!(1234)
  x = zeros(ComplexF64, N, N)
  for i = 1:rank
    x[:, i] = (0.3 + 0.7 * randn()) * cos.(2 * pi / N * rand(1:div(N, 4)) * collect(1:N))
  end
  for i = rank+1:N
    for j = 1:rank
      x[:, i] .+= rand() * x[:, j]
    end
  end
  x = vec(x)

  σ = sum(abs.(x)) / length(x) * σ
  xNoisy = copy(x)
  xNoisy[:] += σ / sqrt(2.0) * (randn(N * N) + 1im * randn(N * N))

  x_lr = copy(xNoisy)
  x_lr = Array(prox!(NuclearRegularization, arrayType(x_lr), 5 * σ, svtShape=(32, 32)))
  @test norm(x - x_lr) <= norm(x - xNoisy)
  @test norm(x - x_lr) / norm(x) ≈ 0 atol = 0.05
  @info "rel. LR error : $(norm(x - x_lr)/ norm(x)) vs $(norm(x - xNoisy)/ norm(x))"
  # check decrease of objective function
  @test 0.5 * norm(xNoisy - x_lr)^2 + norm(NuclearRegularization, x_lr, 5 * σ, svtShape=(N, N)) <= norm(NuclearRegularization, xNoisy, 5 * σ, svtShape=(N, N))
end

function testLLR(shape=(32, 32, 80), blockSize=(4, 4); σ=0.05, arrayType=Array)
  @info "test LLR regularization"
  Random.seed!(1234)
  x = zeros(ComplexF64, shape)
  for j = 1:div(shape[2], blockSize[2])
    for i = 1:div(shape[1], blockSize[1])
      ampl = rand()
      r = rand()
      for t = 1:shape[3]
        x[(i-1)*blockSize[1]+1:i*blockSize[1], (j-1)*blockSize[2]+1:j*blockSize[2], t] .= ampl * exp.(-r * t)
      end
    end
  end
  x = vec(x)

  xNoisy = copy(x)
  σ = sum(abs.(x)) / length(x) * σ
  xNoisy[:] += σ / sqrt(2.0) * (randn(prod(shape)) + 1im * randn(prod(shape)))

  x_llr = copy(xNoisy)
  x_llr = Array(prox!(LLRRegularization, arrayType(x_llr), 10 * σ, shape=shape[1:2], blockSize=blockSize, randshift=false))
  @test norm(x - x_llr) <= norm(x - xNoisy)
  @test norm(x - x_llr) / norm(x) ≈ 0 atol = 0.05
  @info "rel. LLR error : $(norm(x - x_llr)/ norm(x)) vs $(norm(x - xNoisy)/ norm(x))"
  # check decrease of objective function
  @test 0.5 * norm(xNoisy - x_llr)^2 + norm(LLRRegularization, x_llr, 10 * σ, shape=shape[1:2], blockSize=blockSize, randshift=false) <= norm(LLRRegularization, xNoisy, 10 * σ, shape=shape[1:2], blockSize=blockSize, randshift=false)
end

function testLLROverlapping(shape=(32, 32, 80), blockSize=(4, 4); σ=0.05, arrayType=Array)
  @info "test Overlapping LLR regularization"
  Random.seed!(1234)
  x = zeros(ComplexF64, shape)
  for j = 1:div(shape[2], blockSize[2])
    for i = 1:div(shape[1], blockSize[1])
      ampl = rand()
      r = rand()
      for t = 1:shape[3]
        x[(i-1)*blockSize[1]+1:i*blockSize[1], (j-1)*blockSize[2]+1:j*blockSize[2], t] .= ampl * exp.(-r * t)
      end
    end
  end
  x = vec(x)

  xNoisy = copy(x)
  σ = sum(abs.(x)) / length(x) * σ
  xNoisy[:] += σ / sqrt(2.0) * (randn(prod(shape)) + 1im * randn(prod(shape)))

  x_llr = copy(xNoisy)
  prox!(LLRRegularization, x_llr, 10 * σ, shape=shape[1:2], blockSize=blockSize, fullyOverlapping=true)
  @test norm(x - x_llr) <= norm(x - xNoisy)
  @test norm(x - x_llr) / norm(x) ≈ 0 atol = 0.05
  @info "rel. LLR error : $(norm(x - x_llr)/ norm(x)) vs $(norm(x - xNoisy)/ norm(x))"
  # check decrease of objective function
  #@test 0.5*norm(xNoisy-x_llr)^2+normLLR(x_llr,10*σ,shape=shape[1:2],blockSize=blockSize,randshift=false) <= normLLR(xNoisy,10*σ,shape=shape[1:2],blockSize=blockSize,randshift=false)
end

function testLLR_3D(shape=(32, 32, 32, 80), blockSize=(4, 4, 4); σ=0.05, arrayType=Array)
  @info "test LLR 3D regularization"
  Random.seed!(1234)
  x = zeros(ComplexF64, shape)
  for k = 1:div(shape[3], blockSize[3])
    for j = 1:div(shape[2], blockSize[2])
      for i = 1:div(shape[1], blockSize[1])
        ampl = rand()
        r = rand()
        for t = 1:shape[4]
          x[(i-1)*blockSize[1]+1:i*blockSize[1], (j-1)*blockSize[2]+1:j*blockSize[2], (k-1)*blockSize[3]+1:k*blockSize[3], t] .= ampl * exp.(-r * t)
        end
      end
    end
  end
  x = vec(x)

  xNoisy = copy(x)
  σ = sum(abs.(x)) / length(x) * σ
  xNoisy[:] += σ / sqrt(2.0) * (randn(prod(shape)) + 1im * randn(prod(shape)))

  x_llr = copy(xNoisy)
  x_llr = Array(prox!(LLRRegularization, arrayType(x_llr), 10 * σ, shape=shape[1:end-1], blockSize=blockSize, randshift=false))
  @test norm(x - x_llr) <= norm(x - xNoisy)
  @test norm(x - x_llr) / norm(x) ≈ 0 atol = 0.05
  @info "rel. LLR 3D error : $(norm(x - x_llr)/ norm(x)) vs $(norm(x - xNoisy)/ norm(x))"
  # check decrease of objective function # TODO: Implement norm as ND
  # @test 0.5*norm(xNoisy-x_llr)^2+normLLR(x_llr,10*σ,shape=shape[1:3],blockSize=blockSize,randshift=false) <= normLLR(xNoisy,10*σ,shape=shape[1:3],blockSize=blockSize,randshift=false)
end

function testConversion()
  for (xType, lambdaType) in [(Float32, Float64), (Float64, Float32), (Complex{Float32}, Float64), (Complex{Float64}, Float32)]
    for prox in [L1Regularization, L21Regularization, L2Regularization, LLRRegularization, NuclearRegularization, TVRegularization]
      @info "Test λ conversion for prox!($prox, $xType, $lambdaType)"
      @test try
        prox!(prox, zeros(xType, 10), lambdaType(0.0); shape=(2, 5), svtShape=(2, 5))
        true
      catch e
        false
      end skip = in(prox, [LLRRegularization])
      @test try
        norm(prox, zeros(xType, 10), lambdaType(0.0); shape=(2, 5), svtShape=(2, 5))
        true
      catch e
        false
      end skip = in(prox, [LLRRegularization])
    end
  end
end

@testset "Proximal Maps" begin
  @testset "$arrayType" begin
    @testset "L2 Prox" testL2Prox(; arrayType)
    @testset "L1 Prox" testL1Prox(; arrayType)
    @testset "L21 Prox" testL21Prox(; arrayType)
    @testset "TV Prox" testTVprox(; arrayType)
    @testset "TV Prox Directional" testDirectionalTVprox(; arrayType)
    @testset "Positive Prox" testPositive(; arrayType)
    @testset "Projection Prox" testProj(; arrayType)
    @testset "Nuclear Prox" testNuclear(; arrayType)
    @testset "LLR Prox: $arrayType" testLLR(; arrayType)
    @testset "LLR Prox Overlapping: $arrayType" testLLROverlapping(; arrayType)
    @testset "LLR Prox 3D: $arrayType" testLLR_3D(; arrayType)
  end
  @testset "Prox Lambda Conversion" testConversion()
end
