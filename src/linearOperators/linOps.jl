import Base: vcat, size, *, eltype, transpose
import LinearAlgebra.mul!

##############################################################
# light-weight infrastructure needed to model linear operators
##############################################################
abstract type AbstractLinOp{T} end

mutable struct LinOp{T, I<:Integer, F, Ft, Fct, S} <: AbstractLinOp{T}
  nrow::I
  ncol::I
  symmetric::Bool
  hermitian::Bool
  prod!::F
  tprod!::Ft
  ctprod!::Fct
  args5::Bool
  use_prod5!::Bool
  Mv5::S
  Mtu5::S
  allocated5::Bool
end

function LinOp{T}(
  nrow::I,
  ncol::I,
  symmetric::Bool,
  hermitian::Bool,
  prod!::F,
  tprod!::Ft,
  ctprod!::Fct;
  S::DataType = Vector{T},
) where {T, I <: Integer, F, Ft, Fct}

  args5 = (get_nargs(prod!) == 4)
  allocated5 = args5 ? true : false
  use_prod5! = args5 ? true : false

  Mv5, Mtu5 = S(undef, 0), S(undef, 0)
  return LinOp{T, I, F, Ft, Fct, S}(
    nrow,
    ncol,
    symmetric,
    hermitian,
    prod!,
    tprod!,
    ctprod!,
    args5,
    use_prod5!,
    Mv5,
    Mtu5,
    allocated5
  )
end

####################################
# operator properties
####################################
size(op::AbstractLinOp) = (op.nrow, op.ncol)

function size(op::AbstractLinOp, d::Integer)
  nrow, ncol = size(op)
  if d>2
    error("Linear operators only have 2 dimensions for now")
  end
  if d == 1
    return nrow
  end
  return ncol  
end

storage_type(op::LinOp) = typeof(op.Mv5)

eltype(op::AbstractLinOp{T}) where T = T

issymmetric(op::AbstractLinOp) = op.symmetric

has_args5(op) = op.args5
use_prod5!(op::AbstractLinOp) = op.use_prod5!
isallocated5(op::AbstractLinOp) = op.allocated5

get_nargs(f) = first(methods(f)).nargs - 1


function allocate_vectors_args3!(op::AbstractLinOp)
  S = storage_type(op)
  op.Mv5 = S(undef, op.nrow)
  op.Mtu5 = (op.nrow == op.ncol) ? op.Mv5 : S(undef, op.ncol)
  op.allocated5 = true
end

#########
# product
#########
function mul!(res::AbstractVector, op::AbstractLinOp{T}, v::AbstractVector{T}, α, β) where {T}
  use_p5! = use_prod5!(op)
  # allocated storage vectors for 5-args mul if necessary
  has_args5(op) || (β == 0) || isallocated5(op) || allocate_vectors_args3!(op)
  if use_p5!
    op.prod!(res, v, α, β)
  else
    prod3!(res, op.prod!, v, α, β, op.Mv5)
  end
end

function mul!(res::AbstractVector, op::AbstractLinOp{T}, v::AbstractVector{T}) where {T}
  mul!(res, op, v, one(T), zero(T))
end

function prod3!(res, prod!, v, α, β, Mv5)
  if β == 0
    prod!(res, v)
    if α != 1
      res .*= α
    end
  else
    prod!(Mv5, v)
    res .= α .* Mv5 .+ β .* res
  end
end

function *(op::AbstractLinOp{T}, v::AbstractVector{S}) where {T, S}
  nrow = size(op,1)
  res = similar(v, promote_type(T, S), nrow)
  mul!(res, op, v)
  return res
end

#####################
# transposed operator
#####################
struct TransposedLinOp{T, S} <: AbstractLinOp{T}
  parent::S
  function TransposedLinOp{T, S}(A::S) where {T, S}
    new{T,S}(A)
  end
end
TransposedLinOp(A) = TransposedLinOp{eltype(A), typeof(A)}(A)

transpose(A::AbstractLinOp) = TransposedLinOp(A)
transpose(A::TransposedLinOp) = A.parent

size(A::TransposedLinOp) = size(A.parent)[[2; 1]]

issymmetric(op::TransposedLinOp) = issymmetric(op.parent)
has_args5(op::TransposedLinOp) = has_args5(op.parent)
use_prod5!(op::TransposedLinOp) = use_prod5!(op.parent)
isallocated5(op::TransposedLinOp) = isallocated5(op.parent)
allocate_vectors_args3!(op::TransposedLinOp) = allocate_vectors_args3!(op.parent)
storage_type(op::TransposedLinOp) = storage_type(op.parent)

function mul!(
  res::AbstractVector,
  op::TransposedLinOp{T, S},
  v::AbstractVector,
  α,
  β,
) where {T, S}
  p = op.parent
  use_p5! = use_prod5!(p)
  has_args5(op) || (α == 1 && β == 0) || isallocated5(op) || allocate_vectors_args3!(op)
  if issymmetric(p)
    return mul!(res, p, v, α, β)
  end
  if p.tprod! !== nothing
    if use_p5!
      return p.tprod!(res, v, α, β)
    else
      return prod3!(res, p.tprod!, v, α, β, p.Mtu5)
    end
  end
  ctprod! = p.ctprod!
  if p.ctprod! === nothing
    if ishermitian(p)
      ctprod! = p.prod!
    else
      error("RegularizeLeasSquares: unable to infer transpose operator")
    end
  end
  conj!(res)
  if use_p5! 
    ctprod!(res, conj.(v), conj(α), conj(β))
  else
    prod3!(res, ctprod!, conj.(v), conj(α), conj(β), p.Mtu5)
  end
  conj!(res)
end

##############
#concatenation
##############
function vcat_prod!(
  res::AbstractVector,
  A::AbstractLinOp{T},
  B::AbstractLinOp{T},
  Anrow::I,
  nV::I,
  u::AbstractVector,
  α,
  β
) where {T, I <: Integer}
  mul!(view(res, 1:Anrow), A, u, α, β)
  mul!(view(res, (Anrow + 1):nV), B, u, α, β)
end

function vcat_ctprod!(
  res::AbstractVector,
  A::AbstractLinOp{T},
  B::AbstractLinOp{T},
  Anrow::I,
  nV::I,
  v::AbstractVector,
  α,
  β
) where {T, I <: Integer}
  mul!(res, A, view(v, 1:Anrow), α, β)
  mul!(res, B, view(v, (Anrow + 1):nV), α, one(T))
end

function vcat(A::AbstractLinOp{T}, B::AbstractLinOp{T}) where T
  size(A, 2) == size(B, 2) || error("RegularizedLeastSquares: inconsistent column sizes in vcat of LinOps")

  Anrow, Bnrow = size(A, 1), size(B, 1)
  nrow = Anrow + Bnrow
  ncol = size(A, 2)
  # T = promote_type(eltype(A), eltype(B))
  prod! = @closure (res, v, α, β) -> vcat_prod!(res, A, B, Anrow, Anrow + Bnrow, v, α, β)
  tprod! = @closure (res, u, α, β) ->
    vcat_ctprod!(res, transpose(A), transpose(B), Anrow, Anrow + Bnrow, u, α, β)
  ctprod! = @closure (res, w, α, β) ->
    vcat_ctprod!(res, adjoint(A), adjoint(B), Anrow, Anrow + Bnrow, w, α, β)
  args5 = (has_args5(A) && has_args5(B))
  S = promote_type(storage_type(A), storage_type(B))
 
  return CompositeLinOp(T, nrow, ncol, false, false, prod!, tprod!, ctprod!, args5, S = S)
  # LinOp{T}(nrow, ncol, false, false, prod!, tprod!, ctprod!, S=S)
end

function vcat(ops::AbstractLinOp{T}...) where T
  op = ops[1]
  for i = 2:length(ops)
    op = [op; ops[i]]
  end
  return op
end

# create operator from other operators with +, *, vcat,...
function CompositeLinOp(
  T::DataType,
  nrow::I,
  ncol::I,
  symmetric::Bool,
  hermitian::Bool,
  prod!::F,
  tprod!::Ft,
  ctprod!::Fct,
  args5::Bool;
  S::DataType = Vector{T},
) where {I <: Integer, F, Ft, Fct}
  Mv5, Mtu5 = S(undef, 0), S(undef, 0)
  allocated5 = true
  use_prod5! = true

  return LinOp{T, I, F, Ft, Fct, S}(
    nrow,
    ncol,
    symmetric,
    hermitian,
    prod!,
    tprod!,
    ctprod!,
    args5,
    use_prod5!,
    Mv5,
    Mtu5,
    allocated5
  )
end
