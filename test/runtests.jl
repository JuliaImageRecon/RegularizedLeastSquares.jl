using RegularizedLeastSquares, LinearAlgebra
# Packages for testing only
using Random, Test
using FFTW

include("Kaczmarz.jl")
include("testProxMaps.jl")
include("testSolvers.jl")
include("testLinOps.jl")