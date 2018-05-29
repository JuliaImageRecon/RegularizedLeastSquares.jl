export DCTOp

#
# Linear Operator to perform an FFT
#
function DCTOp(T::Type, shape::Tuple)

  return LinearOperator{T}(prod(shape), prod(shape), false, false
            , x->vec(dct(reshape(x,shape)))/sqrt(prod(shape)/2.0)
            , nothing
            , y->vec(idct(reshape(y,shape))) * sqrt(prod(shape)/2.0) )
end
