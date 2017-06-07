using LinearSolver
using Base.Test

srand(3)

@testset "Real Linear Solver" begin

A = rand(3,2);
x = rand(2);
b = A*x;

solvers = linearSolverListReal()

for solver in solvers
  S = createLinearSolver(solver, A, iterations=30)
  x_approx = solve(S,b)
  println("Testing solver $solver ...: $x  == $x_approx")
  @test_approx_eq_eps (norm(x-x_approx)/norm(x)) 0 1e-2
end


end

@testset "Complex Linear Solver" begin

A = rand(3,2)+im*rand(3,2);
x = rand(2)+im*rand(2);
b = A*x;

solvers = linearSolverList()

for solver in solvers
  S = createLinearSolver(solver, A, iterations=300)
  x_approx = solve(S,b)
  println("Testing solver $solver ...: $x  == $x_approx")
  @test_approx_eq_eps (norm(x-x_approx)/norm(x)) 0 1e-2
end


end




# build 4x4 Fourier matrix A=[ 1/sqrt(4)*exp(2*pi*im*j*k/4.0) for j=0:3, k=0:3 ]
