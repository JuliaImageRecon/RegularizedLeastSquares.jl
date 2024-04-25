function RegularizedLeastSquares.tv_restrictMagnitude!(x::vecT) where {T, vecT <: AbstractGPUVector{T}}
  gpu_call(x) do ctx, x_
    i = @linearidx(x_)
    @inbounds x_[i] /= max(1, abs(x_[i]))
    return nothing
  end
end

function RegularizedLeastSquares.tv_linearcomb!(rs::vecT, t3, pq::vecT, t2, pqOld::vecT) where {T, vecT <: AbstractGPUVector{T}}
  gpu_call(rs, t3, pq, t2, pqOld) do ctx, rs_, t3_, pq_, t2_, pqOld_
    i = @linearidx(rs_)
    @inbounds rs_[i] = t3_ * pq_[i] - t2_ * pqOld_[i]
    return nothing
  end
end