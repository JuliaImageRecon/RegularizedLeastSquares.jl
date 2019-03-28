export vectorizePattern, SamplingOp

"""
 idx contains sampling index (for the first dimension) of a multidimensional Array
 of size "shape". Transform this into idx into the corresponding vector index
"""
function vectorizePattern(idx::Array{Int}, shape::Tuple)
  return [ floor(Int,(i-1)/size(idx,1))*shape[1]+idx[i] for i = 1:length(idx) ]
end

"""
  return only the vector elements at positions indicated by pattern.
"""
function SamplingOp(pattern::Array{Int}, shape::Tuple)
  ndims(pattern)>1 ?  idx = vectorizePattern(pattern, shape) : idx = pattern
  return opEye(ComplexF64,length(idx))*opRestriction(idx, prod(shape))
end

function SamplingOp(pattern::Array{Bool})
  return LinearOperator(length(pattern), length(pattern), false, false
                  , x->vec(pattern).*x
                  , nothing
                  , y->vec(pattern).*y )
end

function zeropad(x::Array{T}, pattern::Matrix{Bool}) where T
  y = zeros(T,size(pattern))
  y[pattern] = x[pattern]
  return y
end
