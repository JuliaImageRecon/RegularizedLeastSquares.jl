Random.seed!(1234)

@testset "Real Linear Solver" begin

A = rand(3,2);
x = rand(2);
b = A*x;

solvers = linearSolverListReal()

for solver in solvers
  S = createLinearSolver(solver, A, iterations=100)
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
  S = createLinearSolver(solver, A, iterations=100)
  x_approx = solve(S,b)
  @info "Testing solver $solver ...: $x  == $x_approx"
  @test norm(x - x_approx) / norm(x) ≈ 0 atol=0.1
end


end

@testset "General Convex Solver" begin
# fully sampled operator, image and data
N = 32
F = [ 1. / sqrt(N)*exp(-2. *pi*im*j*k/N) for j=0:N-1, k=0:N-1 ]
x = zeros(N)
for i = 1:3
  x[rand(1:N)] = rand()
end
b = 1. / sqrt(N)*fft(x)

# random undersampling
idx = sort(rand(1:N, 16))
b[idx] .= 0
F[idx,:] .= 0

for solver in ["fista"]
  reg = getRegularization("L1",1.e-3)
  S = createLinearSolver(solver,F,reg; iterations=200)
  x_approx = solve(S, b)
  @info "Testing solver $solver ...: $x  == $x_approx"
  @test norm(x - x_approx) / norm(x) ≈ 0 atol=0.1
end

end
