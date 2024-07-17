module RegularizedLeastSquaresCUDAExt

using RegularizedLeastSquares, RegularizedLeastSquares.LinearAlgebra, RegularizedLeastSquares.LinearOperatorCollection
using CUDA, CUDA.CUSPARSE

include("NormalizedRegularization.jl")

end # module