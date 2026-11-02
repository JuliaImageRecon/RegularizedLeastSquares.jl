using RegularizedLeastSquares, LinearAlgebra, RegularizedLeastSquares.LinearOperatorCollection
# Packages for testing only
using Random, Test
using FFTW, Adapt, JLArrays

isTypeDefined = @isdefined arrayType
arrayType = isTypeDefined ? arrayType : Array

# CGNR is moved to `arrayType` with `Adapt.adapt`; the other solvers are still constructed from
# operators that already live on `arrayType`.
function createTestSolver(solver, A, arrayType; AHA = nothing, kwargs...)
  if solver === CGNR
    S = isnothing(AHA) ? createLinearSolver(solver, A; kwargs...) : createLinearSolver(solver, A; AHA, kwargs...)
    return Adapt.adapt(arrayType, S)
  end
  A = isnothing(A) ? A : arrayType(A)
  return isnothing(AHA) ? createLinearSolver(solver, A; kwargs...) : createLinearSolver(solver, A; AHA = arrayType(AHA), kwargs...)
end

@testset "RegularizedLeastSquares" begin
  include("testCreation.jl")
  include("testKaczmarz.jl")
  include("testProxMaps.jl")
  include("testSolvers.jl")
  include("testRegularization.jl")
  include("testMultiThreading.jl")
  include("testAdapt.jl")
end