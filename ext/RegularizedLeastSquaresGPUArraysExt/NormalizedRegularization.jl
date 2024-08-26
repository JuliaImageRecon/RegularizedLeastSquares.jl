function RegularizedLeastSquares.normalize(::SystemMatrixBasedNormalization, A::AbstractGPUArray, b)
  N = size(A, 2)
  energy = sqrt.(mapreduce(abs2, +, A, dims = 2))
  return norm(energy)^2/N
end

function RegularizedLeastSquares.normalize(::SystemMatrixBasedNormalization, P::ProdOp{T, <:WeightingOp, matT}, b) where {T, matT <: AbstractGPUArray}
  N = size(P, 2)
  A = P.B
  energy = sqrt.(P.A.weights.^2 .* mapreduce(abs2, +, A, dims = 2))
  return norm(energy)^2/N
end