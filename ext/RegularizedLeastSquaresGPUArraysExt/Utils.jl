"""
This function enforces the constraint of a real solution.
"""
function RegularizedLeastSquares.enfReal!(x::arrT) where {N, T<:Complex, arrGPUT <: AbstractGPUArray{T}, arrT <: Union{arrGPUT, SubArray{T, N, arrGPUT}}}
  #Returns x as complex vector with imaginary part set to zero
  x .= broadcast(x) do el
    return complex(real(el))
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
  x .= broadcast(x) do el
    return real(el) < 0 ? im*imag(el) : el
  end
end

"""
This function enforces positivity constraints on its input.
"""
function RegularizedLeastSquares.enfPos!(x::arrT) where {T<:Real, arrT <: AbstractGPUArray{T}}
  #Return x as complex vector with negative parts projected onto 0
  x .= broadcast(x) do el
    return el < 0 ? zero(T) : el
  end
end

RegularizedLeastSquares.rownorm²(A::AbstractGPUMatrix,row::Int64) = sum(map(abs2, @view A[row, :]))
RegularizedLeastSquares.rownorm²(B::Transpose{T,S},row::Int64) where {T,S<:AbstractGPUArray} = sum(map(abs2, @view B.parent[:, row]))

RegularizedLeastSquares.rownorm²(A::ProdOp{T, WeightingOp{T2, vecT2}, matT}, row::Int64) where {T, T2, vecT2 <: AbstractGPUArray, matT} = GPUArrays.@allowscalar A.A.weights[row]^2 * rownorm²(A.B, row)

RegularizedLeastSquares.dot_with_matrix_row(A::AbstractGPUMatrix{T}, x::AbstractGPUVector{T}, k::Int64) where {T} = reduce(+, x .* view(A, k, :))
RegularizedLeastSquares.dot_with_matrix_row(B::Transpose{T,S}, x::V, k::Int64) where {T,S<:AbstractGPUMatrix{T},V<:AbstractGPUVector{T}} = reduce(+, x .* view(B.parent, :, k))
RegularizedLeastSquares.dot_with_matrix_row(B::Transpose{Complex{T},S}, x::V, k::Int64) where {T<:Real,S<:AbstractGPUMatrix{Complex{T}},V<:AbstractGPUVector{Complex{T}}} = reduce(+, x .* view(B.parent, :, k))
RegularizedLeastSquares.dot_with_matrix_row(A::ProdOp{T, WeightingOp{T2, vecT2}, matT}, x::AbstractGPUVector{T}, k::Int64) where {T, T2, vecT2 <: AbstractGPUArray, matT} = GPUArrays.@allowscalar A.A.weights[k] * RegularizedLeastSquares.dot_with_matrix_row(A.B, x, k)