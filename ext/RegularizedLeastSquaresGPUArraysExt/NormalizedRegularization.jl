function RegularizedLeastSquares.normalize(::SystemMatrixBasedNormalization, A::AbstractGPUArray, b)
  N = size(A, 2)
  energy = sqrt.(mapreduce(abs2, +, A, dims = 2))
  return norm(energy)^2/N
end