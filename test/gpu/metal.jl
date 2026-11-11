using Metal
using RegularizedLeastSquares, LinearAlgebra, RegularizedLeastSquares.LinearOperatorCollection
using Random, Test
using FFTW, Adapt, JLArrays

# Metal has no Float64, so runtests.jl cannot run here. This runs the Float32 and ComplexF32
# solvers that are moved to MtlArray with Adapt.adapt part way through and compared with the CPU.
adaptType = MtlArray

# Load the test functions without the Float32/Float64 testsets at the end of the file
include(ex -> Meta.isexpr(ex, :macrocall) ? nothing : ex, joinpath(@__DIR__(), "..", "testSolvers.jl"))

@testset "RegularizedLeastSquares on Metal" begin
  @testset "CGNR resumed after Adapt.adapt" testCGNRAdaptResume(; elType=Float32)
  @testset "FISTA, OptISTA and POGM resumed after Adapt.adapt: $T" for T in (Float32, ComplexF32)
    testProxAdaptResume(; elType=T)
  end
end
