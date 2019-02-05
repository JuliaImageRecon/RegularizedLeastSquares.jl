export DCTOp

#
# Linear Operator to perform a DCT-I
#
function DCTOp(T::Type, shape::Tuple)
  # FIXME: The DCT-I is normally not symmetric...
  return LinearOperator(prod(shape), prod(shape), true, true
            , x->vec(FFTW.r2r(reshape(x,shape),FFTW.REDFT00)) / sqrt(prod(shape)/2.0)
            , nothing
            , nothing )
end
