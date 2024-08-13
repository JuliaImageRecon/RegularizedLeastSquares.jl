# # Computed Tomography Example
# In this example we will go through a simple example from the field of Computed Tomography.
# In addtion to RegularizedLeastSquares.jl, we will need the packages [LinearOperatorCollection.jl](https://github.com/JuliaImageRecon/LinearOperatorCollection.jl), [ImagePhantoms.jl](https://github.com/JuliaImageRecon/ImagePhantoms.jl), [ImageGeoms.jl](https://github.com/JuliaImageRecon/ImageGeoms.jl)
# and [RadonKA.jl](https://github.com/roflmaostc/RadonKA.jl/tree/main), as well as [CairoMakie.jl](https://docs.makie.org/stable/) for visualization. We can install them the same way we did RegularizedLeastSquares.jl.

# RadonKA is a package for the computation of the Radon transform and its adjoint. It is implemented with KernelAbstractions.jl and supports GPU acceleration. See the GPU acceleration [how-to](../howto/gpu_acceleration.md) for more information.

# ## Preparing the Inverse Problem
# To get started, let us generate a simple phantom using the ImagePhantoms and ImageGeom packages:

using ImagePhantoms, ImageGeoms
N = 256
image = shepp_logan(N, SheppLoganToft())
size(image)

# This produces a 64x64 image of a Shepp-Logan phantom. 

using RadonKA, LinearOperatorCollection
angles = collect(range(0, π, 256))
sinogram = Array(RadonKA.radon(image, angles))

# Afterwards we build a Radon operator implementing both the forward and adjoint Radon transform
A = RadonOp(eltype(image); angles, shape = size(image));

# To visualize our image we can use CairoMakie:
using CairoMakie
function plot_image(figPos, img; title = "", width = 150, height = 150)
  ax = CairoMakie.Axis(figPos[1, 1]; yreversed=true, title, width, height)
  hidedecorations!(ax)
  hm = heatmap!(ax, img)
  Colorbar(figPos[2, 1], hm, vertical = false, flipaxis = false)
end
fig = Figure()
plot_image(fig[1,1], image, title = "Image")
plot_image(fig[1,2], sinogram, title = "Sinogram")
plot_image(fig[1,3], backproject(sinogram, angles), title = "Backprojection")
resize_to_layout!(fig)
fig

# ## Solving the Inverse Problem
# To recover the image from the measurement vector, we solve the $l^2_2$-regularized least squares problem
# ```math
# \begin{equation}
#   \underset{\mathbf{x}}{argmin} \frac{1}{2}\vert\vert \mathbf{A}\mathbf{x}-\mathbf{b} \vert\vert_2^2 + \vert\vert\mathbf{x}\vert\vert^2_2 .
# \end{equation}
# ```

# For this purpose we build a $l^2_2$ with regularization parameter $λ=0.001$

using RegularizedLeastSquares
reg = L2Regularization(0.001);

# To solve this inverse problem, the Conjugate Gradient Normal Residual (CGNR) algorithm can be used. Thus, we build the corresponding solver

solver = createLinearSolver(CGNR, A; reg=reg, iterations=20);

# and apply it to our measurement vector
img_approx = solve!(solver, vec(sinogram))

# To visualize the reconstructed image, we need to reshape the result vector to the correct shape. Afterwards we can use CairoMakie again:
img_approx = reshape(img_approx,size(image));
plot_image(fig[1,4], img_approx, title = "Reconstructed Image")
resize_to_layout!(fig)
fig