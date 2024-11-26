# # Regularization
# When formulating inverse problems, a regulariser is formulated as an additional cost function which has to be minimised.
# Many algorithms deal with a regularizers $g$, by computing the proximal map:

# ```math
# \begin{equation}
#   prox_g (\mathbf{x}) = \underset{\mathbf{u}}{argmin} \frac{1}{2}\vert\vert \mathbf{u}-\mathbf{x} \vert {\vert}^2 + g(\mathbf{x}).
# \end{equation}
# ```
# For many regularizers, the proximal map can be computed efficiently in a closed form.

# In order to implement these proximal mappings, RegularizedLeastSquares.jl defines the following type hierarchy:
# ```julia
# abstract type AbstractRegularization
# prox!(reg::AbstractRegularization, x)
# norm(reg::AbstractRegularization, x)
# ```
# Here `prox!(reg, x)` is an in-place function which computes the proximal map on the input-vector `x`. The function `norm` computes the value of the corresponding term in the inverse problem. RegularizedLeastSquares.jl provides `AbstractParameterizedRegularization` and `AbstractProjectionRegularization` as core regularization types.

# ## Parameterized Regularization Terms
# This group of regularization terms features a regularization parameter `λ` that is used during the `prox!` and `norm `computations. Examples of this regulariztion group are `L1`, `L2` or `LLR` (locally low rank) regularization terms.

# These terms are constructed by supplying a `λ` and optionally term specific keyword arguments:
using RegularizedLeastSquares
l2 = L2Regularization(0.3)
# Parameterized regularization terms implement:
# ```julia
# prox!(reg::AbstractParameterizedRegularization, x, λ)
# norm(reg::AbstractParameterizedRegularization, x, λ)
# ```
# where `λ` by default is filled with the value used during construction.

# Invoking `λ` on a parameterized term retrieves its regularization parameter. This can be used in a solver to scale and overwrite the parameter as follows:
prox!(l2, [1.0])
#
param = λ(l2)
prox!(l2, [1.0], param*0.2)


# ## Projection Regularization Terms
# This group of regularization terms implement projections, such as a positivity constraint or a projection with a given convex projection function.
# These are essentially proximal maps where $g(\mathbf{x})$ is the indicator function of a convex set.

positive = PositiveRegularization()
prox!(positive, [2.0, -0.2])

# ## Nested Regularization Terms
# Nested regularization terms are terms that act as decorators to the core regularization terms.
# These terms can be nested around other terms and add functionality to a regularization term, such as scaling `λ` based on the provided operator or applying a transform, such as the Wavelet, to `x`.
# As an example, we can nest a `L1` regularization term around a `Wavelet` operator.

# First we generate an image and apply a wavelet operator to it:
using Wavelets, LinearOperatorCollection, ImagePhantoms, ImageGeoms
N = 256
image = shepp_logan(N, SheppLoganToft())
wop = WaveletOp(Float32, shape = size(image))

wavelet_image = reshape(wop*vec(image), size(image))
wavelet_image = log.(abs.(wavelet_image) .+ 0.01)

# We will use CairoMakie for visualization:
using CairoMakie
function plot_image(figPos, img; title = "", width = 150, height = 150)
  ax = CairoMakie.Axis(figPos; yreversed=true, title, width, height)
  hidedecorations!(ax)
  heatmap!(ax, img)
end
fig = Figure()
plot_image(fig[1,1], image, title = "Image")
plot_image(fig[1,2], wavelet_image, title = "Wavelet Image")
resize_to_layout!(fig)
fig

# To apply soft-thresholding in the wavelet domain, we can nest a `L1` regularization term around the `Wavelet` operator:
core = L1Regularization(0.1)
reg = TransformedRegularization(core, wop);

# We can then apply the proximal map to the image or the image in the wavelet domain:
img_prox_image = prox!(core, copy(vec(image)));
img_prox_wavelet = prox!(reg, copy(vec(image)));

# We can visualize the result:
img_prox_image = reshape(img_prox_image, size(image))
img_prox_wavelet = reshape(img_prox_wavelet, size(image))
plot_image(fig[1,3], img_prox_image, title = "Reg. Image Domain")
plot_image(fig[1,4], img_prox_wavelet, title = "Reg. Wavelet Domain")
resize_to_layout!(fig)
fig

# Generally, regularization terms can be nested arbitrarly deep, adding new functionality with each layer. Each nested regularization term can return its `inner` regularization term.
# Furthermore, all regularization terms implement the iteration interface to iterate over the nesting. The innermost regularization term of a nested term must be a core regularization term and it can be returned by the `sink` function:
RegularizedLeastSquares.innerreg(reg) == core
# 
sink(reg) == core
# 
foreach(r -> println(nameof(typeof(r))), reg)