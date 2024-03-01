using RegularizedLeastSquares, LinearAlgebra, RegularizedLeastSquares.LinearOperatorCollection
# Packages for testing only
using Random, Test
using FFTW

include("testKaczmarz.jl")
include("testProxMaps.jl")
include("testSolvers.jl")