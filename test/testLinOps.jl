function testGradOp1d(N=512)
  x = rand(N)
  G = RegularizedLeastSquares.GradientOp(eltype(x),size(x))
  G0 = Bidiagonal(ones(N),-ones(N-1), :U)[1:N-1,:]

  y = G*x
  y0 = G0*x
  @test norm(y - y0) / norm(y0) ≈ 0 atol=0.001

  xr = transpose(G)*y
  xr0 = transpose(G0)*y0

  @test norm(xr - xr0) / norm(xr0) ≈ 0 atol=0.001
end

function testGradOp2d(N=64)
  x = repeat(1:N,1,N)
  G = RegularizedLeastSquares.GradientOp(eltype(x),size(x))
  G_1d = Bidiagonal(ones(N),-ones(N-1), :U)[1:N-1,:]

  y = G*vec(x)
  y0 = vcat( vec(G_1d*x), vec(x*transpose(G_1d)) )
  @test norm(y - y0) / norm(y0) ≈ 0 atol=0.001

  xr = transpose(G)*y
  y0_x = reshape(y0[1:N*(N-1)],N-1,N)
  y0_y = reshape(y0[N*(N-1)+1:end],N,N-1)
  xr0 = transpose(G_1d)*y0_x + y0_y*G_1d
  xr0 = vec(xr0)

  @test norm(xr - xr0) / norm(xr0) ≈ 0 atol=0.001
end

function testDirectionalGradOp(N=64)
  x = rand(ComplexF64,N,N)
  G1 = RegularizedLeastSquares.GradientOp(eltype(x),size(x),1)
  G2 = RegularizedLeastSquares.GradientOp(eltype(x),size(x),2)
  G_1d = Bidiagonal(ones(N),-ones(N-1), :U)[1:N-1,:]

  y1 = G1*vec(x)
  y2 = G2*vec(x)
  y1_ref = zeros(ComplexF64, N-1,N)
  y2_ref = zeros(ComplexF64, N, N-1)
  for i=1:N
    y1_ref[:,i] .= G_1d*x[:,i]
    y2_ref[i,:] .= G_1d*x[i,:]
  end

  @test norm(y1-vec(y1_ref)) / norm(y1_ref) ≈ 0 atol=0.001
  @test norm(y2-vec(y2_ref)) / norm(y2_ref) ≈ 0 atol=0.001
  
  x1r = transpose(G1)*y1
  x2r = transpose(G2)*y2

  x1r_ref = zeros(ComplexF64, N,N)
  x2r_ref = zeros(ComplexF64, N,N)
  for i=1:N
    x1r_ref[:,i] .= transpose(G_1d)*y1_ref[:,i]
    x2r_ref[i,:] .= transpose(G_1d)*y2_ref[i,:]
  end
  @test norm(x1r-vec(x1r_ref)) / norm(x1r_ref) ≈ 0 atol=0.001
  @test norm(x2r-vec(x2r_ref)) / norm(x2r_ref) ≈ 0 atol=0.001
end

@testset "Linear Operators" begin
  @info "test gradientOp"
  testGradOp1d(512)
  testGradOp2d(64)
  testDirectionalGradOp(64)
end