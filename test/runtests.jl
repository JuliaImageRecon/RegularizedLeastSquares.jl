using RegularizedLeastSquares, LinearAlgebra
# Packages for testing only
using Random, Test
using SparsityOperators: fft

include("Kaczmarz.jl")
include("testProxMaps.jl")
include("testSolvers.jl")