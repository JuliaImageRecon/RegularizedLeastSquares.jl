# # Callbacks
# For certain reconstructions it is important to monitor the internal state of the solver. RegularizedLeastSquares.jl provides a callback mechanism to allow developres to access this state after each iteration.
# The package provides a variety of default callbacks, which for example store the solution after each iteration. More information can be found in the API reference for the `solve!` function.

# In this example we will revist the compressed sensing compressed sensing [example](../examples/compressed_sensing.md) and implement a custom callback using the do-syntax of the `solve!` function.

# We first recreate the operator `A` and the measurement vector `b`:
using ImagePhantoms, ImageGeoms
N = 256
image = shepp_logan(N, SheppLoganToft())
size(image)
using Random, LinearOperatorCollection
randomIndices = shuffle(eachindex(image))
sampledIndices = sort(randomIndices[1:div(end, 3)])
A = SamplingOp(eltype(image), pattern = sampledIndices , shape = size(image));
b = A*vec(image);

# Next we prepare our visualization helper function:
using CairoMakie
function plot_image(figPos, img; title = "", width = 150, height = 150, clim = extrema(img))
  ax = CairoMakie.Axis(figPos; yreversed=true, title, width, height)
  hidedecorations!(ax)
  heatmap!(ax, img, colorrange = clim)
end

# Now we construct the solver with the TV regularization term:
using RegularizedLeastSquares
reg = TVRegularization(0.01; shape=size(image));
solver = createLinearSolver(FISTA, A; reg=reg, iterations=20);

# We will now implement a callback that plots the solution every four iteration:
fig = Figure()
idx = 1
solve!(solver, b) do solver, iteration
  if iteration % 4 == 0
    img_approx = copy(solversolution(solver))
    img_approx = reshape(img_approx, size(image))
    plot_image(fig[div(idx -1, 3) + 1, mod1(idx, 3)], img_approx, clim = extrema(image), title = "$iteration")
    global idx += 1
  end
end
resize_to_layout!(fig)
fig

# In the callback we have to copy the solution, as the solver will update it in place.
# As is explained in the solver section, each features fields that are intended to be immutable during a `solve!` call and a state that is modified in each iteration.
# Depending on the solvers parameters and the measurement input, the state can differ in its fields and their type. Ideally, one tries to avoid accessing the state directly and uses the provided functions to access the state.
