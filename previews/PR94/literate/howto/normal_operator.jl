# # Normal Operator
# Many solvers in RegularizedLeastSquares.jl are based on the normal operator $\mathbf{A}^*\mathbf{A}$ of the linear operator $\mathbf{A}$.
# 
# Solvers such as ADMM, FISTA and POGM generally solve optimization problems of the form:
# ```math
# \begin{equation}
#   \underset{\mathbf{x}}{argmin} \mathbf{f(x)}+ \mathbf{R(x)}
# \end{equation}
# ```
# and require the gradient of the function $\mathbf{f(x)}$. In this package we specialise the function $\mathbf{f(x)}$ to the least squares norm:
# ```math
# \begin{equation}
#   \mathbf{f(x)} = \frac{1}{2}\vert\vert \mathbf{A}\mathbf{x}-\mathbf{b} \vert\vert^2_2
# \end{equation}
# ```
# The gradient of this function is:
# ```math
# \begin{equation}
#   \nabla \mathbf{f(x)} = \mathbf{A}^*(\mathbf{A}\mathbf{x}-\mathbf{b}) = \mathbf{A}^*\mathbf{Ax} - \mathbf{A}^*\mathbf{b}
# \end{equation}
# ```
# Similarily, the conjugate gradient normal residual (CGNR) algorithm applies conjugate gradient algorithm to:
# ```math
# \begin{equation}
#   \mathbf{A}^*\mathbf{A}\mathbf{x} = \mathbf{A}^*\mathbf{b}
# \end{equation}
# ```
# The normal operator can be passed directly to these solvers, otherwise it is computed internally.
using RegularizedLeastSquares
A = randn(32, 16)
x = randn(16)
b = A*x

solver = createLinearSolver(CGNR, A; AHA = adjoint(A) * A, reg = L2Regularization(0.0001), iterations=32);
x_approx = solve!(solver, b)

# The normal operator can also be computed using the `normalOperator` function from LinearOperatorCollection.jl. This is useful if the normal operator is not directly available or shouldn't be stored in memory.
# This function is opinionated and attempts to optimize the resulting operator for iterative applications. Specifying a custom method for a custom operator allows one to control this optimization.

# An example of such an optimization is a matrix-free weighting of $\mathbf{A}$ as shown in the [Weighting](weighting.md) example:
using LinearOperatorCollection
weights = rand(32)
WA = ProdOp(WeightingOp(weights), A)
AHA = LinearOperatorCollection.normalOperator(WA)

# Without an optimization a matrix-free product would apply the following operator each iteration:
# ```math
# \begin{equation}
#   (\mathbf{WA})^*\mathbf{WA} = \mathbf{A}^*\mathbf{W}^*\mathbf{W}\mathbf{A}
# \end{equation}
# ```
# This is not efficient and instead the normal operator can be optimized by initially computing the weights:
# ```math
# \begin{equation}
#   \tilde{\mathbf{W}} = \mathbf{W}^*\mathbf{W}
# \end{equation}
# ```
# and then applying the following each iteration:
# ```math
# \begin{equation}
#   \mathbf{A}^*\tilde{\mathbf{W}}\mathbf{A}
# \end{equation}
# ```

# The optimized normal operator can then be passed to the solver:
solver = createLinearSolver(CGNR, WA; AHA = AHA, reg = L2Regularization(0.0001), iterations=32);
x_approx2 = solve!(solver, weights .* b)
# Of course it is also possible to optimize a normal operator with other means and pass it to the solver via the AHA keyword argument.

# It is also possible to only supply the normal operator to these solvers, however on then needs to supply $\mathbf{A^*b}$ intead of $\mathbf{b}$.