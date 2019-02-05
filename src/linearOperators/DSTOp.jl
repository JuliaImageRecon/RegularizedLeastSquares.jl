export DSTOp

#
# Linear Operator to perform a DST
#
function DSTOp(T::Type, shape::Tuple)

  return LinearOperator(prod(shape), prod(shape), true, false
            , x->vec(FFTW.r2r(reshape(x,shape),FFTW.RODFT10)).*weights(shape)
            , nothing
            , y->vec(FFTW.r2r(reshape(y ./ weights(shape) ,shape),FFTW.RODFT01)) ./ (8*prod(shape))  )
end

function weights(s)
  w = ones(s...)./sqrt(8*prod(s))
  w[s[1],:,:]./=sqrt(2)
  w[:,s[2],:]./=sqrt(2)
  w[:,:,s[3]]./=sqrt(2)
  return reshape(w,prod(s))
end
