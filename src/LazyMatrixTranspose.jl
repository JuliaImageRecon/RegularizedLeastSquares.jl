# This will hopefully at some time be supported by Julia Base

import Base: size, getindex
export ctransp, transp, MatrixTranspose

struct MatrixTranspose{T, S, Conj} <: AbstractArray{T, 2}
 data::S
end

ctransp(A::AbstractArray{T, 2}) where {T} = MatrixTranspose{T, typeof(A), true}(A)
transp(A::AbstractArray{T, 2}) where {T<:Real} = MatrixTranspose{T, typeof(A), true}(A)
transp(A::AbstractArray{T, 2}) where {T} = MatrixTranspose{T, typeof(A), false}(A)

size(A::MatrixTranspose) = reverse(size(A.data))
size(A::MatrixTranspose, dim::Integer) = dim == 1 ? size(A.data, 2) : (dim == 2 ? size(A.data, 1) : size(A.data, dim))

getindex(A::MatrixTranspose, i::Integer, j::Integer) = getindex(A.data, j, i)
