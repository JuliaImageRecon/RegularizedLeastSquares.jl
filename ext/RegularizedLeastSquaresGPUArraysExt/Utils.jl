"""
This function enforces the constraint of a real solution.
"""
function RegularizedLeastSquares.enfReal!(x::arrT) where {N, T<:Complex, arrGPUT <: AbstractGPUArray{T}, arrT <: Union{arrGPUT, SubArray{T, N, arrGPUT}}}
  #Returns x as complex vector with imaginary part set to zero
  gpu_call(x) do ctx, x_
    i = @linearidx(x_)
    @inbounds (x_[i] = complex(x_[i].re))
    return nothing
  end
end

"""
This function enforces the constraint of a real solution.
"""
RegularizedLeastSquares.enfReal!(x::arrT) where {N, T<:Real, arrGPUT <: AbstractGPUArray{T}, arrT <: Union{arrGPUT, SubArray{T, N, arrGPUT}}} = nothing

"""
This function enforces positivity constraints on its input.
"""
function RegularizedLeastSquares.enfPos!(x::arrT) where {N, T<:Complex, arrGPUT <: AbstractGPUArray{T}, arrT <: Union{arrGPUT, SubArray{T, N, arrGPUT}}}
  #Return x as complex vector with negative parts projected onto 0
  gpu_call(x) do ctx, x_
    i = @linearidx(x_)
    @inbounds (x_[i].re < 0) && (x_[i] = im*x_[i].im)
    return nothing
  end
end

"""
This function enforces positivity constraints on its input.
"""
function RegularizedLeastSquares.enfPos!(x::arrT) where {T<:Real, arrT <: AbstractGPUArray{T}}
  #Return x as complex vector with negative parts projected onto 0
  gpu_call(x) do ctx, x_
    i = @linearidx(x_)
    @inbounds (x_[i] < 0) && (x_[i] = zero(T))
    return nothing
  end
end

RegularizedLeastSquares.rownorm²(A::AbstractGPUMatrix,row::Int64) = sum(map(abs2, @view A[row, :]))
RegularizedLeastSquares.rownorm²(B::Transpose{T,S},row::Int64) where {T,S<:AbstractGPUArray} = sum(map(abs2, @view B.parent[:, row]))


RegularizedLeastSquares.dot_with_matrix_row(A::AbstractGPUMatrix{T}, x::AbstractGPUVector{T}, k::Int64) where {T} = reduce(+, x .* view(A, k, :))
RegularizedLeastSquares.dot_with_matrix_row(B::Transpose{T,S}, x::AbstractGPUVector{T}, k::Int64) where {T,S<:AbstractGPUArray} = reduce(+, x .* view(B.parent, :, k))