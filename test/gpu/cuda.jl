using CUDA

arrayType = CuArray

include(joinpath(@__DIR__(), "..", "runtests.jl"))