module RegularizedLeastSquaresGPUArraysExt

using RegularizedLeastSquares, RegularizedLeastSquares.LinearAlgebra, RegularizedLeastSquares.LinearOperatorCollection
using GPUArrays, KernelAbstractions

include("Utils.jl")
include("ProxTV.jl")
include("ProxL21.jl")
include("ProxLLR.jl")
include("NormalizedRegularization.jl")
include("Kaczmarz.jl")

end