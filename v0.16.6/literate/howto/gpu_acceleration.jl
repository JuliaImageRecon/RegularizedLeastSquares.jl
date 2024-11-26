# # GPU Acceleration
# RegularizedLeastSquares.jl supports generic GPU acceleration. This means that the user can use any GPU array type that supports the GPUArrays interface. This includes [CUDA.jl](https://github.com/JuliaGPU/CUDA.jl), [AMDGPU.jl](https://github.com/JuliaGPU/AMDGPU.jl), and [Metal.jl](https://github.com/JuliaGPU/Metal.jl).
# In this example we will use the package JLArrays.jl which provides a reference implementation for GPUArrays, that can runs on CPUs.
using JLArrays
gpu = JLArray;

# To use the following examples on an actual GPU, load the appropraite package replace `JLArray` with the respective GPU array type, for example:
# ```julia
# using CUDA
# gpu = CuArray
# ```

# At first we will look at an example of dense GPU arrays.
using RegularizedLeastSquares
A = gpu(rand(Float32, 32, 16))
x = gpu(rand(Float32, 16))
b = A*x;

# Solvers adapt their states based on the type of the given measurement vector. This means that the solver will automatically switch to GPU acceleration if a GPU array is passed as the measurement vector.
solver = createLinearSolver(CGNR, A; reg = L2Regularization(0.0001), iterations=32);
x_approx = solve!(solver, b)

# This adaption does not include the operator. So if we want to compare with CPU result, we need to construct a new solver with a CPU operator.
solver = createLinearSolver(CGNR, Array(A); reg = L2Regularization(0.0001), iterations=32);
x_cpu = solve!(solver, Array(b))
isapprox(Array(x_approx), x_cpu)

# ## Matrix-Free Operators
# A special case is the usage of matrix-free operators. Since these operators do not have a concrete matrix representation, their GPU support depends on their implementation.
# Since not all multiplications within a solver approximation are in-place, the operator also needs to support the `*` operation and construct an appropriate result vector.
# For matrix-free operators based on LinearOperators.jl, this can be achieved by implementing the `LinearOperators.storage_type` method.

# In the following, we will take another look at the CS example and execute it on the GPU. Note that for the JLArray example we chose a small phantom, since the JLArray implementation is not optimized for performance:
using ImagePhantoms, ImageGeoms
N = 32
image = shepp_logan(N, SheppLoganToft())

using Random, LinearOperatorCollection
randomIndices = shuffle(eachindex(image))
sampledIndices = sort(randomIndices[1:div(end, 3)]);

# To construct the operator, we need to convert the indices to a GPU array.
# We also need to specify the correct storage type. In both LinearOperators.jl and LinearOperatorCollection.jl this is done with the `S` keyword argument.
gpu_indices = gpu(sampledIndices)
A = SamplingOp(eltype(image), pattern = gpu_indices, shape = size(image), S = typeof(b));

# Let's inspect the storage type of the operator:
using LinearOperatorCollection.LinearOperators
LinearOperators.storage_type(A)

# Afterwards we can use the operator as usual:
b = A*vec(gpu(image));

# And use it in the solver:
using RegularizedLeastSquares
reg = TVRegularization(0.01; shape=size(image))
solver = createLinearSolver(FISTA, A; reg=reg, iterations=20)
img_approx = solve!(solver,b);

# To visualize the reconstructed image, we need to reshape the result vector to the correct shape and convert it to a CPU array:
img_approx = reshape(Array(img_approx),size(image))

# We will again use CairoMakie for visualization:
using CairoMakie
function plot_image(figPos, img; title = "", width = 150, height = 150)
  ax = CairoMakie.Axis(figPos; yreversed=true, title, width, height)
  hidedecorations!(ax)
  heatmap!(ax, img)
end
fig = Figure()
plot_image(fig[1,1], image, title = "Image")
samplingMask = fill(false, size(image))
samplingMask[sampledIndices] .= true
plot_image(fig[1,2], image .* samplingMask, title = "Sampled Image")
plot_image(fig[1,3], img_approx, title = "Reconstructed Image")
resize_to_layout!(fig)
fig