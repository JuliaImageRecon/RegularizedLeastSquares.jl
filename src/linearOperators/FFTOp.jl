export FFTOp, \

mutable struct FFTOp{T,F1<:FuncOrNothing,F2<:FuncOrNothing,F3<:FuncOrNothing} <: AbstractLinearOperator{T,F1,F2,F3}
  nrow   :: Int
  ncol   :: Int
  symmetric :: Bool
  hermitian :: Bool
  prod   :: F1 # apply the operator to a vector
  tprod  :: F2 # apply the transpose operator to a vector
  ctprod :: F3 # apply the transpose conjugate operator to a vector
end

"""
  FFTOp(T::Type, shape::Tuple, shift=true)

returns an operator which performs an FFT on Arrays of type T

# Arguments:
* `T::Type`       - type of the array to transform
* `shape::Tuple`  - size of the array to transform
* (`shift=true`)  - if true, fftshifts are performed
"""
function FFTOp(T::Type, shape::Tuple, shift=true)
  plan = plan_fft(zeros(T, shape);flags=FFTW.MEASURE)
  iplan = plan_ifft(zeros(T, shape);flags=FFTW.MEASURE)

  if shift
    return FFTOp{T,Function, Nothing,Function}(prod(shape), prod(shape), false, false
              , x->vec(fftshift(plan*fftshift(reshape(x,shape))))/sqrt(prod(shape))
              , nothing
              , y->vec(ifftshift(iplan*ifftshift(reshape(y,shape)))) * sqrt(prod(shape)) )
  else
    return FFTOp{T,Function, Nothing,Function}(prod(shape), prod(shape), false, false
            , x->vec(plan*(reshape(x,shape)))/sqrt(prod(shape))
            , nothing
            , y->vec(iplan*(reshape(y,shape))) * sqrt(prod(shape)) )
  end
end

\(A::FFTOp, x::Vector) = adjoint(A) * x
