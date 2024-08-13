# # Getting Started

# In this example we will go through a simple example from the field of Compressed Sensing to get familiar with RegularizedLeastSquares.jl.

# ## Installation

# To install RegularizedLeastSquares.jl, you can use the Julia package manager. Open a Julia REPL and run the following command:

# ```julia
# using Pkg
# Pkg.add("RegularizedLeastSquares")
# ```
# This will download and install the RegularizedLeastSquares.jl package and its dependencies. To install a different version, please consult the [Pkg documentation](https://pkgdocs.julialang.org/dev/managing-packages/#Adding-packages). 

# Once the installation is complete, you can import the package in your code with the `using` keyword:


using RegularizedLeastSquares

# RegularizedLeastSquares aims to solve inverse problems of the form:
# ```math
# \begin{equation}
#   \underset{\mathbf{x}}{argmin} \frac{1}{2}\vert\vert \mathbf{A}\mathbf{x}-\mathbf{b} \vert\vert_2^2 + \mathbf{R(x)} .
# \end{equation}
# ```
# where $\mathbf{A}$ is a linear operator, $\mathbf{y}$ is the measurement vector, and $\mathbf{R(x)}$ is an (optional) regularization term.
# The aim is to reconstruct the unknown vector $\mathbf{x}$. In this first exampel we will just reconstruct a simple random matrix. For more advanced examples, please refer to the examples.

A = rand(32, 16)
x = rand(16)
b = A*x;

# To approximate x from b, we can use the Conjugate Gradient Normal Residual (CGNR) algorithm. We first build the corresponding solver:
solver = createLinearSolver(CGNR, A; iterations=32);

# and apply it to our measurement vector
x_approx = solve!(solver, b)
isapprox(x, x_approx, rtol = 0.001)

# Usually the inverse problems are ill-posed and require regularization. RegularizedLeastSquares.jl provides a variety of regularization terms.
# The CGNR algorithm can solve optimzation problems of the form:
# ```math
# \begin{equation}
#   \underset{\mathbf{x}}{argmin} \frac{1}{2}\vert\vert \mathbf{A}\mathbf{x}-\mathbf{b} \vert\vert_2^2 + \vert\vert\mathbf{x}\vert\vert^2_2 .
# \end{equation}
# ```

# The corresponding solver can be built with the L2 regularization term:
solver = createLinearSolver(CGNR, A; reg = L2Regularization(0.0001), iterations=32);
x_approx = solve!(solver, b)
isapprox(x, x_approx, rtol = 0.001)
