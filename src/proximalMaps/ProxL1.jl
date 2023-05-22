export L1Regularization, proxL1!, proxL1, normL1

struct L1Regularization <: AbstractRegularization
  λ::Float64
  sparseTrafo::Trafo
end
L1Regularization(λ; sparseTrafo::Trafo=nothing, kargs...) = L1Regularization(λ, sparseTrafo)

prox!(reg::L1Regularization, x, λ) = proxL1!(x, λ; sparseTrafo = reg.sparseTrafo)
norm(reg::L1Regularization, x, λ) = normL1(x, λ; sparseTrafo = reg.sparseTrafo)


"""
    proxL1!(x::Array{T}, λ::Float64; sparseTrafo::Trafo=nothing, kargs...) where T

performs soft-thresholding - i.e. proximal map for the Lasso problem.

# Arguments:
* `x::Array{T}`                 - Vector to apply proximal map to
* `λ::Float64`                  - regularization paramter
* `sparseTrafo::Trafo=nothing`  - sparsifying transform to apply
"""
function proxL1!(x::AbstractArray{Tc}, λ::T; sparseTrafo::Trafo=nothing, kargs...) where {T, Tc <: Union{T, Complex{T}}}
  ε = eps(T)

  if sparseTrafo != nothing
    z = sparseTrafo*x
    z .= max.((abs.(z).-λ),0) .* (z.+ε)./(abs.(z).+ε)
    x .= adjoint(sparseTrafo)*z
  else
    x .= max.((abs.(x).-λ),0) .* (x.+ε)./(abs.(x).+ε)
  end

  return x
end

"""
    normL1(x::Array{T}, λ::Float64; sparseTrafo::Trafo=nothing, kargs...) where T

returns the value of the L1-regularization term.
Arguments are the same as in `proxL1!`
"""
function normL1(x::T, λ::Float64; sparseTrafo::Trafo=nothing, kargs...) where T<:AbstractArray
  if sparseTrafo != nothing
    l1Norm = λ*norm(sparseTrafo*x,1)
  else
    l1Norm = λ*norm(x,1)
  end
  return l1Norm
end
