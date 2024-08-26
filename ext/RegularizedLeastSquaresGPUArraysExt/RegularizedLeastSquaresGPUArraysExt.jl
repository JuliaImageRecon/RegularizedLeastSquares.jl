module RegularizedLeastSquaresGPUArraysExt

using RegularizedLeastSquares, RegularizedLeastSquares.LinearAlgebra, RegularizedLeastSquares.LinearOperatorCollection
using GPUArrays

include("Utils.jl")
include("ProxTV.jl")
include("ProxL21.jl")
include("NormalizedRegularization.jl")
include("Kaczmarz.jl")

end