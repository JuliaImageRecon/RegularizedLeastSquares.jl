function testDCT1d(N=32)
  Random.seed!(1235)
  x = zeros(ComplexF64, N^2)
  for i=1:5
    x .+= rand()*cos.(rand(1:N^2)*collect(1:N^2)) .+ 1im*rand()*cos.(rand(1:N^2)*collect(1:N^2))
  end
  D1 = DCTOp(Float64,(N^2,),2)
  D2 = sqrt(2/N^2)*[cos(pi/(N^2)*j*(k+0.5)) for j=0:N^2-1,k=0:N^2-1]
  D2[1,:] .*= 1/sqrt(2)
  D3 = DCTOp(Float64,(N^2,),4)
  D4 = sqrt(2/N^2)*[cos(pi/(N^2)*(j+0.5)*(k+0.5)) for j=0:N^2-1,k=0:N^2-1]

  y1 = D1*x
  y2 = D2*x

  @test norm(y1 - y2) / norm(y1) ≈ 0 atol=0.01

  y3 = D3*x
  y4 = D4*x
  @test norm(y3 - y4) / norm(y1) ≈ 0 atol=0.01

  x1 = adjoint(D1)*y1
  @test norm(x1 - x) / norm(x) ≈ 0 atol=0.01

  x2 = adjoint(D3)*y3
  @test norm(x2 - x) / norm(x) ≈ 0 atol=0.01
end

function testFFT1d(N=32,shift=true)
  Random.seed!(1234)
  x = zeros(N^2)
  for i=1:5
    x .+= rand()*cos.(rand(1:N^2)*collect(1:N^2))
  end
  D1 = FFTOp(Float64,(N^2,),shift)
  D2 =  1.0/N*[exp(-2*pi*im*j*k/N^2) for j=0:N^2-1,k=0:N^2-1]

  y1 = D1*x
  if shift
    y2 = fftshift(D2*fftshift(x))
  else
    y2 = D2*x
  end
  @test norm(y1 - y2) / norm(y1) ≈ 0 atol=0.01

  x1 = adjoint(D1) * y1
  if shift
    x2 = ifftshift(adjoint(D2)*ifftshift(y2))
  else
    x2 = adjoint(D2)*y2
  end
  @test norm(x1 - x2) / norm(x1) ≈ 0 atol=0.01
end

function testFFT2d(N=32,shift=true)
  Random.seed!(1234)
  x = zeros(N^2)
  for i=1:5
    x .+= rand()*cos.(rand(1:N^2)*collect(1:N^2))
  end
  D1 = FFTOp(Float64,(N,N),shift)

  idx = CartesianIndices((N,N))[collect(1:N^2)]
  D2 =  1.0/N*[ exp(-2*pi*im*((idx[j][1]-1)*(idx[k][1]-1)+(idx[j][2]-1)*(idx[k][2]-1))/N) for j=1:N^2, k=1:N^2 ]

  y1 = D1*x
  if shift
    y2 = D2*vec(fftshift(reshape(x,N,N)))
    y2 = vec(fftshift(reshape(y2,N,N)))
  else
    y2 = D2*x
  end
  @test norm(y1 - y2) / norm(y1) ≈ 0 atol=0.01

  x1 = adjoint(D1) * y1
  if shift
    x2 = adjoint(D2)*vec(ifftshift(reshape(y2,N,N)))
    x2 = vec(ifftshift(reshape(x2,N,N)))
  else
    x2 = adjoint(D2)*y2
  end
  @test norm(x1 - x2) / norm(x1) ≈ 0 atol=0.01
end

function testWeighting(N=512)
  Random.seed!(1234)
  x1 = rand(N)
  weights = rand(N)
  W = WeightingOp(weights)
  y1 = W*x1
  y = weights .* x1

  @test norm(y1 - y) / norm(y) ≈ 0 atol=0.01

  x2 = rand(2*N)
  W2 = WeightingOp(weights,2)
  y2 = W2*x2
  y = repeat(weights,2) .* x2

  @test norm(y2 - y) / norm(y) ≈ 0 atol=0.01
end

@testset "Linear Operators" begin
  @info "test DCT-II and DCT-IV"
  for N in [2,8,16,32]
    testDCT1d(N)
  end
  @info "test FFT"
  for N in [8,16,32]
    testFFT1d(N,false)
    testFFT1d(N,true)
    testFFT2d(N,false)
    testFFT2d(N,true)
  end
  @info "test Weighting"
  testWeighting(512)
end
