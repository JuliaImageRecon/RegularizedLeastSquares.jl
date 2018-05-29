using Wavelets
export WaveletOp


function WaveletOp(shape, wt=wavelet(WT.db2))
  return LinearOperator{Complex128}(maximum(shape)^2, prod(shape), false, false
            , x->waveletProd(x,shape,wt)
            , nothing
            , y->waveletCTProd(y,shape,wt) )
end

function waveletProd{T}(x::Vector{T},shape, wt)
  if shape[1] != shape[2]
    xSquare = zeros(T, maximum(shape), maximum(shape))
    xSquare[1:shape[1],1:shape[2]] = reshape(x,shape)
  else
    xSquare = reshape(x,shape)
  end
  return vec( dwt(xSquare, wt) )
end

function waveletCTProd{T}(y::Vector{T},shape, wt)
  squareSize = (maximum(shape), maximum(shape))
  x = idwt( reshape(y, squareSize), wt)
  return vec( x[1:shape[1],1:shape[2]] )
end
