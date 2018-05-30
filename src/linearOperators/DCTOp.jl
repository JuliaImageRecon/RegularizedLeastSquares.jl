export DCTOp

#
# Linear Operator to perform an FFT
#
function DCTOp(T::Type, shape::Tuple)

  return LinearOperator{T}(prod(shape), prod(shape), true, true
            , x->vec(FFTW.r2r(reshape(x,shape),FFTW.REDFT00)) / sqrt(prod(shape)/2.0)
            , nothing
            , nothing )
end
