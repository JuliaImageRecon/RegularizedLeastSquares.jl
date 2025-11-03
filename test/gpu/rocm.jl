using AMDGPU

arrayType = ROCArray

include(joinpath(@__DIR__(), "..", "runtests.jl"))