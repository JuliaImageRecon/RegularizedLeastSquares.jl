function RegularizedLeastSquares.tv_restrictMagnitude!(x::vecT) where {T, vecT <: AbstractGPUVector{T}}
  @kernel inbounds = true cpu = false function tv_restrict_kernel(x)
    i = @index(Global, Linear)
    x[i] /= max(1, abs(x[i]))
  end
  kernel! = tv_restrict_kernel(get_backend(x))
  kernel!(x, ndrange = length(x))
end

function RegularizedLeastSquares.tv_linearcomb!(rs::vecT, t3, pq::vecT, t2, pqOld::vecT) where {T, vecT <: AbstractGPUVector{T}}
  @kernel inbounds = true cpu = false function tv_linearcomb_kernel(rs, t3, pq, t2, pqOld)
    i = @index(Global, Linear)
    rs[i] = t3 * pq[i] - t2 * pqOld[i]
  end
  kernel! = tv_linearcomb_kernel(get_backend(rs))
  kernel!(rs, t3, pq, t2, pqOld, ndrange = length(rs))
end