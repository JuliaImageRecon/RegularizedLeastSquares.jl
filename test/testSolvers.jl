Random.seed!(1234)

@testset "Real Linear Solver" begin
  A = rand(3,2);
  x = rand(2);
  b = A*x;

  solvers = linearSolverListReal()

  for solver in solvers
    solverInfo = SolverInfo()
    S = createLinearSolver(solver, A, iterations=100, solverInfo=solverInfo)
    x_approx = solve(S,b)
    @info "Testing solver $solver ...: $x  == $x_approx"
    @test norm(x - x_approx) / norm(x) ≈ 0 atol=0.1
  end
end

@testset "Complex Linear Solver" begin
  A = rand(3,2)+im*rand(3,2);
  x = rand(2)+im*rand(2);
  b = A*x;

  solvers = linearSolverList()

  for solver in solvers
    solverInfo = SolverInfo()
    S = createLinearSolver(solver, A, iterations=100, solverInfo=solverInfo)
    x_approx = solve(S,b)
    @info "Testing solver $solver ...: $x  == $x_approx"
    @test norm(x - x_approx) / norm(x) ≈ 0 atol=0.1
  end
end

@testset "General Convex Solver" begin
  # fully sampled operator, image and data
  N = 256
  numPeaks = 5
  F = [ 1. / sqrt(N)*exp(-2. *pi*im*j*k/N) for j=0:N-1, k=0:N-1 ]
  x = zeros(N)
  for i = 1:3
    x[rand(1:N)] = rand()
  end
  b = 1. / sqrt(N)*fft(x)

  # random undersampling
  idx = sort(unique(rand(1:N, div(N,2))))
  b = b[idx]
  F = F[idx,:]

  for solver in ["fista","admm"]
    reg = Regularization("L1",1.e-3)
    solverInfo = SolverInfo()
    S = createLinearSolver(solver,F; reg=reg, iterations=200, solverInfo=solverInfo)
    x_approx = solve(S, b)
    @info "Testing solver $solver ...: relative error = $(norm(x - x_approx) / norm(x))"
    @test norm(x - x_approx) / norm(x) ≈ 0 atol=0.1
  end
  for solver in ["splitBregman"]
    reg = Regularization("L1",1.e-3)
    solverInfo = SolverInfo()
    S = createLinearSolver(solver,F; reg=reg,iterations=5,iterationsInner=40,μ=1.e3,
                                     λ=1.e3,ρ=0.0,solverInfo=solverInfo)
    x_approx = solve(S, b)
    @info "Testing solver $solver ...: relative error = $(norm(x - x_approx) / norm(x))"
    @test norm(x - x_approx) / norm(x) ≈ 0 atol=0.1
  end
end
