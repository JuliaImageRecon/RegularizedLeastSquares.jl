using RegularizedLeastSquares, LinearAlgebra, RegularizedLeastSquares.LinearOperatorCollection
# Packages for testing only
using Random, Test
using FFTW, Adapt, JLArrays

isTypeDefined = @isdefined arrayType
arrayType = isTypeDefined ? arrayType : Array
# Backend that the Adapt tests move CPU-built solvers to; on the CPU we use JLArrays as a stand-in GPU array type
adaptType = arrayType == Array ? JLArray : arrayType

@testset "RegularizedLeastSquares" begin
  include("testCreation.jl")
  include("testKaczmarz.jl")
  include("testProxMaps.jl")
  include("testSolvers.jl")
  include("testRegularization.jl")
  include("testMultiThreading.jl")
end