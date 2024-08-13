# # Compressed Sensing Example
# In this example we will go through a simple example from the field of Compressed Sensing.
# In addtion to RegularizedLeastSquares.jl, we will need the packages [LinearOperatorCollection.jl](https://github.com/JuliaImageRecon/LinearOperatorCollection.jl), [ImagePhantoms.jl](https://github.com/JuliaImageRecon/ImagePhantoms.jl), [ImageGeoms.jl](https://github.com/JuliaImageRecon/ImageGeoms.jl) and Random.jl, as well as [CairoMakie.jl](https://docs.makie.org/stable/) for visualization. We can install them the same way we did RegularizedLeastSquares.jl.

# ## Preparing the Inverse Problem
# To get started, let us generate a simple phantom using the ImagePhantoms and ImageGeom packages:

using ImagePhantoms, ImageGeoms
N = 256
image = shepp_logan(N, SheppLoganToft())
size(image)

# This produces a 256x256 image of a Shepp-Logan phantom. 

# In this example, we consider a problem in which we randomly sample a third of the pixels in the image. Such a problem and the corresponding measurement can be constructed with the packages LinearOperatorCollection and Random:

# We first randomly shuffle the indices of the image and then select the first third of the indices to sample.
using Random, LinearOperatorCollection
randomIndices = shuffle(eachindex(image))
sampledIndices = sort(randomIndices[1:div(end, 3)])

# Afterwards we build a sampling operator which samples the image at the selected indices
A = SamplingOp(eltype(image), pattern = sampledIndices , shape = size(image));

# Then we apply the sampling operator to the image to obtain the measurement vector
b = A*vec(image);

# To visualize our image we can use CairoMakie:
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
resize_to_layout!(fig)
fig

# ## Solving the Inverse Problem
# To recover the image from the measurement vector, we solve the TV-regularized least squares problem
# ```math
# \begin{equation}
#   \underset{\mathbf{x}}{argmin} \frac{1}{2}\vert\vert \mathbf{A}\mathbf{x}-\mathbf{b} \vert\vert_2^2 + \vert\vert\mathbf{x}\vert\vert_{\lambda\text{TV}} .
# \end{equation}
# ```

# For this purpose we build a TV regularizer with regularization parameter $λ=0.01$

using RegularizedLeastSquares
reg = TVRegularization(0.01; shape=size(image));

# To solve this CS problem, the Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) can be used. Thus, we build the corresponding solver

solver = createLinearSolver(FISTA, A; reg=reg, iterations=20);

# and apply it to our measurement vector
img_approx = solve!(solver,b)

# To visualize the reconstructed image, we need to reshape the result vector to the correct shape. Afterwards we can use CairoMakie again:
img_approx = reshape(img_approx,size(image));
plot_image(fig[1,3], img_approx, title = "Reconstructed Image")
resize_to_layout!(fig)
fig