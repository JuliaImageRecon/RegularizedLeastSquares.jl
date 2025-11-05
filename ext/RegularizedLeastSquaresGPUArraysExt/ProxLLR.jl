function RegularizedLeastSquares.proxLLRNonOverlapping!(reg::LLRRegularization{TR, N, TI}, x::Union{AbstractGPUArray{T}, AbstractGPUArray{Complex{T}}}, λ::T) where {TR, N, TI, T}
    x_cpu = Array(x)
    RegularizedLeastSquares.proxLLRNonOverlapping!(reg, x_cpu, λ)
    copyto!(x, x_cpu)
    return x
end