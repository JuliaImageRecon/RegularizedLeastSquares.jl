# # Plug-and-Play Regularization
# A group of regularization terms that can not be directly written down as function are learned plug-and-play (PnP) priors.
# These are terms based on deep neural networks, which are trainted to implement the proximal map corresponding to the regularization term.
# Such a PnP prior can be used in the same way as any other regularization term.

# The following example shows how to use a PnP prior in the context of the `Kaczmarz` solver.
using RegularizedLeastSquares
A = randn(32, 16)
x = randn(16)
b = A*x;

# For the documentation we will just use the identity function as a placeholder for the PnP prior.
model = identity

# In practice, you would replace this with a neural network:
# ```julia
# using Flux
# model = Flux.loadmodel!(model, ...)
# ```

# The model can then be used together with the `PnPRegularization` term:
reg = PnPRegularization(1.0; model = model, shape = [16]);

# Since models often expect a specific input range, we can use the `MinMaxTransform` to normalize the input:
reg = PnPRegularization(1.0; model = model, shape = [16], input_transform = RegularizedLeastSquares.MinMaxTransform);
# Custom input transforms can be implemented by passing something callable as the `input_transform` keyword argument. For more details see the PnPRegularization documentation.

# The regularization term can then be used in the solver:
solver = createLinearSolver(Kaczmarz, A; reg = reg, iterations = 32)
x_approx = solve!(solver, b)
