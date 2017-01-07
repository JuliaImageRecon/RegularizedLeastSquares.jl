# This will hopefully at some time be supported by Julia Base

import Base: size, getindex
export ctransp, transp, MatrixTranspose

immutable MatrixTranspose{T, S, Conj} <: AbstractArray{T, 2}
 data::S
end

ctransp{T}(A::AbstractArray{T, 2}) = MatrixTranspose{T, typeof(A), true}(A)
transp{T<:Real}(A::AbstractArray{T, 2}) = MatrixTranspose{T, typeof(A), true}(A)
transp{T}(A::AbstractArray{T, 2}) = MatrixTranspose{T, typeof(A), false}(A)

size(A::MatrixTranspose) = reverse(size(A.data))
size(A::MatrixTranspose, dim::Integer) = dim == 1 ? size(A.data, 2) : (dim == 2 ? size(A.data, 1) : size(A.data, dim))

getindex(A::MatrixTranspose, i::Integer, j::Integer) = getindex(A.data, j, i)

