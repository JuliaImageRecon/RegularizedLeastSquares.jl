using Wavelets
export WaveletOp

mutable struct WaveletOp{T,F1<:FuncOrNothing,F2<:FuncOrNothing,F3<:FuncOrNothing} <: AbstractLinearOperator{T,F1,F2,F3}
  nrow   :: Int
  ncol   :: Int
  symmetric :: Bool
  hermitian :: Bool
  prod   :: F1 # apply the operator to a vector
  tprod  :: F2 # apply the transpose operator to a vector
  ctprod :: F3 # apply the transpose conjugate operator to a vector
end

"""
  WaveletOp(shape, wt=wavelet(WT.db2))

returns a `ẀaveletOp <: AbstractLinearOperator`, which performs a Wavelet transform on
a given input array.

# Arguments

* `shape`                 - size of the Array to transform
* (`wt=wavelet(WT.db2)`)  - Wavelet to apply
"""
function WaveletOp(shape, wt=wavelet(WT.db2))
  return WaveletOp{Float64,Function,Nothing,Function}(maximum(shape)^2, prod(shape), false, false
            , x->waveletProd(x,shape,wt)
            , nothing
            , y->waveletCTProd(y,shape,wt) )
end

function waveletProd(x::Vector{T},shape, wt) where T
  if shape[1] != shape[2]
    xSquare = zeros(T, maximum(shape), maximum(shape))
    xSquare[1:shape[1],1:shape[2]] = reshape(x,shape)
  else
    xSquare = reshape(x,shape)
  end
  return vec( dwt(xSquare, wt) )
end

function waveletCTProd(y::Vector{T},shape, wt) where T
  squareSize = (maximum(shape), maximum(shape))
  x = idwt( reshape(y, squareSize), wt)
  return vec( x[1:shape[1],1:shape[2]] )
end

\(A::WaveletOp, x::Vector) = adjoint(A) * x
