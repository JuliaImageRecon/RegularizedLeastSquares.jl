# # Efficient Kaczmarz
# Unlike many of the other solvers provided by RegularizedLeastSquares.jl, the Kaczmarz method does not utilize a matrix-vector product with the operator $\mathbf{A}$ nor the normal operator $\mathbf{A*A}$. Instead, it uses the rows of $\mathbf{A}$ to update the solution iteratively.
# Efficient Kaczmarz implementation therefore require very efficient dot products with the rows of $\mathbf{A}$. In RegularizedLeastSquares.jl, this is achieved with the `dot_with_matrix_row` function.
using RegularizedLeastSquares
A = randn(256, 256)
x = randn(256)
b = A*x;

# The `dot_with_matrix_row` function calculates the dot product between a row of A and the current approximate solution of x:
row = 1
isapprox(RegularizedLeastSquares.dot_with_matrix_row(A, x, row), sum(A[row, :] .* x))

# Since in Julia, dense arrays are stored in column-major order, such a row-based operation is quite inefficient. A workaround is to transpose the matrix then pass it to a Kaczmarz solver.
At = collect(transpose(A))
A_eff = transpose(At)

# Note that the transpose function can return a lazy transpose object, so we first collect the transpose into a dense matrix.
# Then we transpose it again to get the efficient representation of the matrix.

# We can compare the performance using the BenchmarkTools.jl package. First for the original matrix:
using BenchmarkTools
solver = createLinearSolver(Kaczmarz, A; reg = L2Regularization(0.0001), iterations=100)
@benchmark solve!(solver, b) samples = 100

# And then for the efficient matrix:
solver_eff = createLinearSolver(Kaczmarz, A_eff; reg = L2Regularization(0.0001), iterations=100)
@benchmark solve!(solver_eff, b) samples = 100

# We can also combine the efficient matrix with a weighting matrix, as is shown in the [Weighting](weighting.md) example.

# Custom operators need to implement the `dot_with_matrix_row` function to be used with the Kaczmarz solver. Ideally, such an implementation is allocation free.