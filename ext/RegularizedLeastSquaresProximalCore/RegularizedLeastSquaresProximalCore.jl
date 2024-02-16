module RegularizedLeastSquaresProximalCore

using RegularizedLeastSquares, ProximalCore

import RegularizedLeastSquares.prox!, RegularizedLeastSquares.ProximalCoreAdapter

struct ProximalCoreAdapterImpl{T, F} <: ProximalCoreAdapter{T, F}
  λ::T
  op::F
end

RegularizedLeastSquares.ProximalCoreAdapter(λ::T, op::F) where {T, F} = ProximalCoreAdapterImpl(λ, op)

function prox!(reg::ProximalCoreAdapter, x::AbstractArray{Tc}, λ::T) where {T, Tc <: Union{T, Complex{T}}}
  ProximalCore.prox!(x, reg.op, x, λ)
  return x
end

end