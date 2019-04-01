using RegularizedLeastSquares, LinearAlgebra, FFTW
# Packages for tetsing only
using Random, Test

include("testOperators.jl")
include("testProxMaps.jl")
include("testSolvers.jl")
