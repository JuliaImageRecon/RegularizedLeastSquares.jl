export DCTOp, \

mutable struct DCTOp{T,F1<:FuncOrNothing,F2<:FuncOrNothing,F3<:FuncOrNothing} <: AbstractLinearOperator{T,F1,F2,F3}
  nrow   :: Int
  ncol   :: Int
  symmetric :: Bool
  hermitian :: Bool
  prod   :: F1 # apply the operator to a vector
  tprod  :: F2 # apply the transpose operator to a vector
  ctprod :: F3 # apply the transpose conjugate operator to a vector
end

"""
  DCTOp(T::Type, shape::Tuple, dcttype=2)

returns a `DCTOp <: AbstractLinearOperator` which performs a DCT on a given input array.

# Arguments:
* `T::Type`       - type of the array to transform
* `shape::Tuple`  - size of the array to transform
* `dcttype`       - type of DCT (currently `2` and `4` are supported)
"""
function DCTOp(T::Type, shape::Tuple, dcttype=2)
  if dcttype == 2
    return DCTOp{T,Function,Nothing,Function}(prod(shape), prod(shape), true, false
            , x->vec((dct(reshape(x,shape))))
            , nothing
            , y->vec((idct(reshape(y,shape)))))
  elseif dcttype == 4
    factor = sqrt(1.0/(prod(shape)* 2^length(shape)) )
    return DCTOp{T,Function,Function,Nothing}(prod(shape), prod(shape), true, false
            , x->vec((FFTW.r2r(reshape(x,shape),FFTW.REDFT11).*factor))
            , x->vec((FFTW.r2r(reshape(x,shape),FFTW.REDFT11).*factor))
            , nothing )
  else
    error("DCT type $(dcttype) not supported")
  end
end

\(A::DCTOp, x::Vector) = adjoint(A) * x
