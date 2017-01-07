#module LinearOperator

import Base: *,\, A_mul_B!, At_mul_B!, size #does not exactly fit

export FFTOperator, SparseFFTOperator, MatrixProduct, SparseDCTOperator, linearOperator, RealBasisTrafo, ComplexBasisTrafo, BasisTrafo, DCTOperator

abstract BasisTrafo
abstract ComplexBasisTrafo <: BasisTrafo
abstract RealBasisTrafo <: BasisTrafo

size(A::BasisTrafo,i::Int) = prod(A.shape)

linearOperator(op::Void,shape) = nothing

function linearOperator(op::AbstractString, shape)
  if op == "FFT"
    trafo = FFTOperator(shape)
  elseif op == "DCT"
    trafo = DCTOperator(shape)
  else
    error("Unknown transformation")
  end
  trafo
end

type FFTOperator  <: ComplexBasisTrafo
  shape
end


### FFTOperator

#FFTOperator(shape) = FFTOperator(shape)

*(A::FFTOperator, x::AbstractArray) = vec(fft(reshape(x, A.shape...))) / sqrt(prod(A.shape))
\(A::FFTOperator, x::AbstractArray) = vec(ifft(reshape(x, A.shape...))) * sqrt(prod(A.shape))

#there semms to be no inplace version for fft. Maybe due to the fact that type is not preserved if real array is passed
function A_mul_B!{T}(A::FFTOperator, x::DenseArray{Complex{T}})
  fft!(reshape(x, A.shape...))
  α = convert(T,1.0 / sqrt(prod(A.shape)))
  scale!(x,α)
end

#there semms to be no inplace version for fft. Maybe due to the fact that type is not preserved if real array is passed
function At_mul_B!{T}(A::FFTOperator, x::DenseArray{Complex{T}})
  ifft!(reshape(x, A.shape...))
  α = convert(T,sqrt(prod(A.shape)))
  scale!(x,α)
end

### DCTOperator

type DCTOperator <: RealBasisTrafo
  shape
end

#DCTOperator(shape) = DCTOperator(shape)

*(A::DCTOperator, x::AbstractArray) = vec(dct(reshape(x, A.shape...))) / sqrt(prod(A.shape)/2.0)
\(A::DCTOperator, x::AbstractArray) = vec(idct(reshape(x, A.shape...))) * sqrt(prod(A.shape)/2.0)

function A_mul_B!{T}(A::DCTOperator, x::DenseArray{T})
  dct!(reshape(x, A.shape...))
  α = convert(T, 1.0 / sqrt(prod(A.shape)/2.0))
  scale!(x,α)
end

function At_mul_B!{T}(A::DCTOperator, x::DenseArray{T})
  idct!(reshape(x, A.shape...))
  α::T = convert(T,sqrt(prod(A.shape) / 2.0))
  scale!(x,α)
end

### SparseFFTOperator

#TODO -> inplace

type SparseFFTOperator{T} <: ComplexBasisTrafo 
  shape::Tuple
  indices::Vector{Int64}
  buffer::Vector{T}
end

function SparseFFTOperator(T::Type, shape::Tuple, indices::Vector{Int64})
        SparseFFTOperator(shape, indices, zeros(T, prod(shape)) )
end

function *{T}(A::SparseFFTOperator, x::Vector{T})
  fft(reshape(x, A.shape))[A.indices] / sqrt(prod(A.shape))
end

function \{T}(A::SparseFFTOperator, x::Vector{T})
  A.buffer[:] = 0.0
  A.buffer[A.indices] = x
  return ifft(reshape(A.buffer, A.shape))[:] * sqrt(prod(A.shape))
end

### SparseDCTOperator

#TODO -> inplace

type SparseDCTOperator{T} <: RealBasisTrafo
  shape::Tuple
  indices::Vector{Int64}
  buffer::Vector{T}
end

function SparseDCTOperator(T::Type, shape::Tuple, indices::Vector{Int64})
        SparseDCTOperator(shape, indices, zeros(T, prod(shape)) )
end

function *{T}(A::SparseDCTOperator, x::Vector{T})
  dct(reshape(x, A.shape))[A.indices] / sqrt(prod(A.shape)/2.0)
end

function \{T}(A::SparseDCTOperator, x::Vector{T})
  A.buffer[:] = 0.0
  A.buffer[A.indices] = x
  return idct(reshape(A.buffer, A.shape))[:] * sqrt(prod(A.shape)/2.0)
end


### MatrixProduct

type MatrixProduct{T} <: AbstractMatrix{T} #LinearOperator
  leftMatrix
  rightMatrix
end

function MatrixProduct{U,T}(leftMatrix::AbstractMatrix{T}, rightMatrix::AbstractMatrix{U})
  MatrixProduct{promote_type(U,T)}(leftMatrix::AbstractMatrix{T}, rightMatrix::AbstractMatrix{U})

end


*{T,U}(A::MatrixProduct{T}, x::Vector{U}) = A.leftMatrix * (A.rightMatrix * x)



function test()
  x = [1.,2.,3.,4.]
  A = FFTOperator((2,2))

  y =  A \ (A * x)

  println(y)

  B = SparseFFTOperator(Complex64, (2,2), [1,2,3])

  y =  B \ (B * x)

  println(y)

  C = SparseDCTOperator(Complex64, (2,2), [1,2,3])

  y =  C \ (C * x)

  println(y)
end

# end # module




