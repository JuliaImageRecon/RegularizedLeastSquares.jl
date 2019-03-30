function testDCT1d(N=32)
  Random.seed!(1234)
  x = zeros(N^2)
  for i=1:5
    x .+= rand()*cos.(rand(1:N^2)*collect(1:N^2))
  end
  D1 = DCTOp(Float64,(N^2,))
  D2 = zeros(N^2,N^2)
  D2[:,1]  .= 0.5
  D2[:,end] .= 0.5*[(-1)^k for k=0:N^2-1]
  D2[:,2:end-1] .=  [cos(pi/(N^2-1)*j*k) for j=0:N^2-1,k=1:N^2-2]
  D2 = 2*sqrt(2.0/N^2)*D2

  y1 = D1*x
  y2 = D2*x
  @test norm(y1 - y2) / norm(y1) ≈ 0 atol=0.01

  x1 = adjoint(D1)*x
  x2 = adjoint(D2)*x # D2*x # FIXME: adjoint(D2)*x would be correct
  @test norm(x1 - x2) / norm(x1) ≈ 0 atol=0.01
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

@testset "Linear Operators" begin
  @info "test DCT-I"
  for N in [8,16,32]
    testDCT1d(N)
  end
  @info "test FFT"
  for N in [8,16,32]
    testFFT1d(N,false)
    testFFT1d(N,true)
    testFFT2d(N,false)
    testFFT2d(N,true)
  end

end
