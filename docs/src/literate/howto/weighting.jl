# # Weighting
# Often time one wants to solve a weighted least squares problem of the form:
# ```math
# \begin{equation}
#   \underset{\mathbf{x}}{argmin} \frac{1}{2}\vert\vert \mathbf{A}\mathbf{x}-\mathbf{b} \vert\vert^2_{\mathbf{W}} + \mathbf{R(x)} .
# \end{equation}
# ```
# where $\mathbf{W}$ is a symmetric, positive weighting matrix and $\vert\vert\mathbf{y}\vert\vert^2_\mathbf{W}$ denotes the weighted Euclidean norm.
# An example of such a weighting matrix is a noise whitening matrix. Another example could be a scaling of the matrix rows by the reciprocal of their row energy.

# In the following, we will solve a weighted least squares problem of the form:
# ```math
# \begin{equation}
#   \underset{\mathbf{x}}{argmin} \frac{1}{2}\vert\vert \mathbf{A}\mathbf{x}-\mathbf{b} \vert\vert_\mathbf{W}^2 + \vert\vert\mathbf{x}\vert\vert^2_2 .
# \end{equation}
# ```
using RegularizedLeastSquares, LinearOperatorCollection, LinearAlgebra
A = rand(32, 16)
x = rand(16)
b = A*x
weights = map(row -> 1/rownorm²(A, row), 1:size(A, 1))
WA = diagm(weights) * A
solver = createLinearSolver(Kaczmarz, WA; reg = L2Regularization(0.0001), iterations=10)
x_approx = solve!(solver, weights .* b);

# The operator A is not always a dense matrix and the product between the operator and the weighting matrix is not always efficient or possible.
# The package [LinearOperatorCollection.jl](https://github.com/JuliaImageRecon/LinearOperatorCollection.jl) provides a matrix-free implementation of a diagonal weighting matrix, as well as a matrix free product between two matrices.
# This weighted operator has efficient implementations of the normal operator and also for the row-action operations of the Kaczmarz solver.
W = WeightingOp(weights)
P = ProdOp(W, A)
solver = createLinearSolver(Kaczmarz, P; reg = L2Regularization(0.0001), iterations=10)
x_approx2 = solve!(solver, W * b)
isapprox(x_approx, x_approx2)