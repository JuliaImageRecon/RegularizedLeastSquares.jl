module RegularizedLeastSquaresCUDAExt

using RegularizedLeastSquares, RegularizedLeastSquares.LinearAlgebra, RegularizedLeastSquares.LinearOperatorCollection
using CUDA, CUDA.CUSPARSE

include("NormalizedRegularization.jl")
include("ProxLLR.jl")

end # module