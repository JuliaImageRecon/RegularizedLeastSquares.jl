using LinearOperators

include("linearOperators/FFTOp.jl")
include("linearOperators/DCTOp.jl")
include("linearOperators/DSTOp.jl")
include("linearOperators/WaveletOp.jl")

export linearOperator, linearOperatorList

linearOperator(op::Void,shape) = nothing

function linearOperatorList()
  return ["DCT", "Cheb", "FFT"]
end

function linearOperator(op::AbstractString, shape)
  if op == "FFT"
    trafo = FFTOp(Complex64, tuple(shape...), false) #FFTOperator(shape)
  elseif op == "DCT"
    trafo = DCTOp(Complex64, tuple(shape...))
  elseif op == "DST"
    trafo = DSTOp(Complex64, tuple(shape...))
  elseif op == "Wavelet"
    trafo = WaveletOp(tuple(shape...))
  else
    error("Unknown transformation")
  end
  trafo
end
