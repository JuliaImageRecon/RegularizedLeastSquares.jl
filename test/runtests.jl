using RegularizedLeastSquares, LinearAlgebra, RegularizedLeastSquares.LinearOperatorCollection
# Packages for testing only
using Random, Test
using FFTW

@testset "RegularizedLeastSquares" begin
  include("testCreation.jl")
  include("testKaczmarz.jl")
  include("testProxMaps.jl")
  include("testSolvers.jl")
  include("testRegularization.jl")
end