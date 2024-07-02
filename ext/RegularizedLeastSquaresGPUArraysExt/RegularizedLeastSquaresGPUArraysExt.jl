module RegularizedLeastSquaresGPUArraysExt

using RegularizedLeastSquares, RegularizedLeastSquares.LinearAlgebra, GPUArrays

include("Utils.jl")
include("ProxTV.jl")
include("ProxL21.jl")
include("Kaczmarz.jl")

end