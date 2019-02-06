# check Thikonov proximal map
function testL2Prox(N=1024; numPeaks=5, λ=0.01)
  @info "test L2-regularization"
  Random.seed!(1234)
  x = zeros(N)
  for i=1:numPeaks
    x[rand(1:N)] = rand()
  end

  x_l2 = 1. / (1. + 2. *λ)*x
  x_l2 = copy(x)
  proxL2!(x_l2,λ)
  @test norm(x_l2 - 1.0/(1.0+2.0*λ)*x) / norm(1.0/(1.0+2.0*λ)*x) ≈ 0 atol=0.001
  # check decrease of objective function
  @test 0.5*norm(x-x_l2)^2 + normL2(x_l2,λ) <= normL2(x,λ)
end

# denoise a signal consisting of a number of delta peaks
function testL1Prox(N=1024; numPeaks=5, σ=0.03)
  @info "test L1-regularization"
  Random.seed!(1234)
  x = zeros(N)
  for i=1:numPeaks
    x[rand(1:N)] = (1-2*σ)*rand()+2*σ
  end

  σ = sum(abs.(x))/length(x)*σ
  xNoisy = x .+ σ/sqrt(2.0)*(randn(N)+1im*randn(N))

  x_l1 = copy(xNoisy)
  proxL1!(x_l1, 2*σ)

  # solution should be better then without denoising
  @info "rel. L1 error : $(norm(x - x_l1)/ norm(x)) vs $(norm(x - xNoisy)/ norm(x))"
  @test norm(x - x_l1) <= norm(x - xNoisy)
  @test norm(x - x_l1) / norm(x) ≈ 0 atol=0.1
  # check decrease of ojective function
  @test 0.5*norm(xNoisy-x_l1)^2+normL1(x_l1,2*σ) <= normL1(xNoisy,2*σ)
end

# denoise a signal consisting  of multiple slices with delta peaks at the same locations
# only the last slices are noisy.
# Thus, the first slices serve as a reference to inhance denoising
function testL21Prox(N=1024; numPeaks=5, numSlices=8, noisySlices=2, σ=0.05)
  @info "test L21-regularization"
  Random.seed!(1234)
  x = zeros(ComplexF64,N,numSlices)
  for i=1:numPeaks
    x[rand(1:N),:] = (1-2*σ)*rand(numSlices) .+ 2*σ
  end
  x = vec(x)

  xNoisy = copy(x)
  noise = randn(N*noisySlices)
  σ = sum(abs.(x))/length(x)*σ
  xNoisy[(numSlices-noisySlices)*N+1:end] .+= σ/sqrt(2.0)*(randn(N*noisySlices)+1im*randn(N*noisySlices)) #noise

  x_l1 = copy(xNoisy)
  proxL1!(x_l1, 2*σ)

  x_l21 = copy(xNoisy)
  proxL21!(x_l21, 2*σ,slices=numSlices)

  # solution should be better then without denoising and with l1-denoising
  @info "rel. L21 error : $(norm(x - x_l21)/ norm(x)) vs $(norm(x - xNoisy)/ norm(x))"
  @test norm(x - x_l21) <= norm(x - xNoisy)
  @test norm(x - x_l21) <= norm(x - x_l1)
  @test norm(x - x_l21) / norm(x) ≈ 0 atol=0.05
  # check decrease of objective function
  @test 0.5*norm(xNoisy-x_l21)^2+normL21(x_l21,2*σ,slices=numSlices) <= normL21(xNoisy,2*σ,slices=numSlices)
  @test 0.5*norm(xNoisy-x_l21)^2+normL21(x_l21,2*σ,slices=numSlices) <= 0.5*norm(xNoisy-x_l1)^2+normL21(x_l1,2*σ,slices=numSlices)
end

# denoise a piece-wise constant signal using TV regularization
function testTVprox(N=1024; numEdges=5, σ=0.05)
  @info "test TV-regularization"
  Random.seed!(1234)
  x = zeros(ComplexF64,N,N)
  for i=1:numEdges
    idx1 = rand(0:N-1)
    idx2 = rand(0:N-1)
    x[idx1+1:end, idx2+1:end] .+= randn()
  end
  x= vec(x)

  xNoisy = copy(x)
  σ = sum(abs.(x))/length(x)*σ
  xNoisy[:] += σ/sqrt(2.0)*(randn(N*N)+1im*randn(N*N))

  x_l1 = copy(xNoisy)
  proxL1!(x_l1, 2*σ)

  x_tv = copy(xNoisy)
  proxTV!(x_tv, 2*σ, shape=(N,N))

  @info "rel. TV error : $(norm(x - x_tv)/ norm(x)) vs $(norm(x - xNoisy)/ norm(x))"
  @test norm(x - x_tv) <= norm(x - xNoisy)
  @test norm(x - x_tv) <= norm(x - x_l1)
  @test norm(x - x_tv) / norm(x) ≈ 0 atol=0.05
  # check decrease of objective function
  @test 0.5*norm(xNoisy-x_tv)^2+normTV(x_tv,2*σ,shape=(N,N)) <= normTV(xNoisy,2*σ,shape=(N,N))
  @test 0.5*norm(xNoisy-x_tv)^2+normTV(x_tv,2*σ,shape=(N,N)) <= 0.5*norm(xNoisy-x_l1)^2+normTV(x_l1,2*σ,shape=(N,N))
end

# test enforcement of positivity constraint
function testPositive(N=1024)
  @info "test positivity-constraint"
  Random.seed!(1234)
  x = randn(N) .+ 1im*randn(N)
  xPos = real.(x)
  xPos[findall(x->x<0,xPos)] .= 0
  xProj = copy(x)
  proxPositive!(xProj)

  @test norm(xProj-xPos)/norm(xPos) ≈ 0 atol=1.e-4
  # check decrease of objective function
  @test 0.5*norm(x-xProj)^2+normPositive(xProj) <= normPositive(x)
end

# test enforcement of "realness"-constraint
function testProj(N=1012)
  @info "test realness-constraint"
  Random.seed!(1234)
  x = randn(N) .+ 1im*randn(N)
  xReal = real.(x)
  xProj = copy(x)
  proxProj!(xProj,0.0,projFunc=x->real(x))
  @test norm(xProj-xReal)/norm(xReal) ≈ 0 atol=1.e-4
  # check decrease of objective function
  @test 0.5*norm(x-xProj)^2+normProj(xProj,projFunc=x->real(x)) <= normProj(x,projFunc=x->real(x))
end

# test denoising of a low-rank matrix
function testNuclear(N=32,rank=2;σ=0.05)
  @info "test nuclear norm regularization"
  Random.seed!(1234)
  x = zeros(ComplexF64,N,N);
  for i=1:rank
    x[:,i] = (0.3+0.7*randn())*cos.(2*pi/N*rand(1:div(N,4))*collect(1:N));
  end
  for i=rank+1:N
    for j=1:rank
      x[:,i] .+= rand()*x[:,j];
    end
  end
  x = vec(x)

  σ = sum(abs.(x))/length(x)*σ
  xNoisy = copy(x)
  xNoisy[:] += σ/sqrt(2.0)*(randn(N*N)+1im*randn(N*N))

  x_lr = copy(xNoisy)
  proxNuclear!(x_lr,5*σ,svtShape=(32,32))
  @test norm(x - x_lr) <= norm(x - xNoisy)
  @test norm(x - x_lr) / norm(x) ≈ 0 atol=0.05
  @info "rel. LR error : $(norm(x - x_lr)/ norm(x)) vs $(norm(x - xNoisy)/ norm(x))"
  # check decreas of objective function
  @test 0.5*norm(xNoisy-x_lr)^2+normNuclear(x_lr,5*σ,svtShape=(N,N)) <= normNuclear(xNoisy,5*σ,svtShape=(N,N))
end

function testLLR(shape=(32,32,80),blockSize=[4,4];σ=0.05)
  @info "test LLR regularization"
  Random.seed!(1234)
  x = zeros(ComplexF64,shape);
  for j=1:div(shape[2],blockSize[2])
    for i=1:div(shape[1],blockSize[1])
      ampl = rand()
      r = rand()
      for t=1:shape[3]
        x[(i-1)*blockSize[1]+1:i*blockSize[1],(j-1)*blockSize[2]+1:j*blockSize[2],t] .= ampl*exp.(-r*t)
      end
    end
  end
  x = vec(x)

  xNoisy = copy(x)
  σ = sum(abs.(x))/length(x)*σ
  xNoisy[:] += σ/sqrt(2.0)*(randn(prod(shape))+1im*randn(prod(shape)))

  x_llr = copy(xNoisy)
  proxLLR!(x_llr,10*σ,shape=shape[1:2],blockSize=blockSize,randshift=false)
  @test norm(x - x_llr) <= norm(x - xNoisy)
  @test norm(x - x_llr) / norm(x) ≈ 0 atol=0.05
  @info "rel. LLR error : $(norm(x - x_llr)/ norm(x)) vs $(norm(x - xNoisy)/ norm(x))"
  # check decreas of objective function
  @test 0.5*norm(xNoisy-x_llr)^2+normLLR(x_llr,10*σ,shape=shape[1:2],blockSize=blockSize,randshift=false) <= normLLR(xNoisy,10*σ,shape=shape[1:2],blockSize=blockSize,randshift=false)
end

@testset "Proximal Maps" begin
  testL2Prox()
  testL1Prox()
  testL21Prox()
  testTVprox()
  testPositive()
  testProj()
  testNuclear()
  testLLR()
end
