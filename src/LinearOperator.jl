using LinearOperators
import Base: \

include("linearOperators/FFTOp.jl")
include("linearOperators/DCTOp.jl")
include("linearOperators/DSTOp.jl")
include("linearOperators/WaveletOp.jl")
include("linearOperators/SamplingOp.jl")

export linearOperator, linearOperatorList

linearOperator(op::Nothing,shape) = nothing

function linearOperatorList()
  return ["DCT-II", "DCT-IV", "FFT"]
end

function linearOperator(op::AbstractString, shape)
  shape_ = tuple(shape...)
  if op == "FFT"
    trafo = FFTOp(ComplexF32, shape_, false) #FFTOperator(shape)
  elseif op == "DCT-II"
    shape_ = tuple(shape[shape .!= 1]...)
    trafo = DCTOp(ComplexF32, shape_, 2)
  elseif op == "DCT-IV"
    shape_ = tuple(shape[shape .!= 1]...)
    trafo = DCTOp(ComplexF32, shape_, 4)
  elseif op == "DST"
    trafo = DSTOp(ComplexF32, shape_)
  elseif op == "Wavelet"
    trafo = WaveletOp(shape_)
  else
    error("Unknown transformation")
  end
  trafo
end
