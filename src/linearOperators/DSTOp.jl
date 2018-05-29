export DSTOp

#
# Linear Operator to perform an FFT
#
function DSTOp(T::Type, shape::Tuple)

  return LinearOperator{T}(prod(shape), prod(shape), true, false
            , x->vec(FFTW.r2r(reshape(x,shape),FFTW.RODFT10)).*weights(shape)
            , nothing
            , y->vec(FFTW.r2r(reshape(y ./ weights(shape) ,shape),FFTW.RODFT01)) ./ (8*prod(shape))  )
end
