export Regularization, lambdList, prox! #, norm

abstract type AbstractRegularization end

"""
Type describing Regularizers

# Fields
* `prox!::Function`           - proximal map for the regularizer
* `norm::Function`            - (semi-)norm for the regularizer
* `λ::Float64`                - regularization paramter
* `params::Dict{Symbol,Any}`  - additional parameters
"""
mutable struct Regularization <: AbstractRegularization
  prox!::Function
  norm::Function
  λ::Float64
  params::Dict{Symbol,Any}  # @TODO in die funcs
end

Base.vec(reg::Regularization) = [reg]
Base.vec(reg::Vector{Regularization}) = reg

function Regularization(;prox!::Function = x->x, norm::Function = norm0,
                         λ::AbstractFloat=0.0, params=Dict{Symbol,Any}())
  Regularization(prox!,norm,λ,params)
end

"""
    RegularizationList()

Returns a list of all available Regularizations
"""
function RegularizationList()
  Any["L2", "L1", "L21", "TV", "LLR", "Positive", "Proj", "Nuclear"]
end

"""
    Regularization(name::String, λ::AbstractFloat; kargs...)

create a Regularization object containing all the infos necessary to calculate a proximal map.

# valid names
* `"L2"`
* `"L1"`
* `"L21"`
* `"TV"`
* `"LLR"`
* `"Nuclear"`
* `"Positive"`
* `"Proj"`
"""
function Regularization(name::String, λ::AbstractFloat; kargs...)
  if name=="L2"
    return Regularization(proxL2!, normL2, λ, kargs)
  elseif name=="L1"
    return Regularization(proxL1!, normL1, λ, kargs)
  elseif name=="L21"
    return Regularization(proxL21!, normL21, λ, kargs)
  elseif name=="TV"
    return Regularization(proxTV!, normTV, λ, kargs)
  elseif name=="LLR"
    return Regularization(proxLLR!, normLLR, λ, kargs)
  elseif name=="Nuclear"
    return Regularization(proxNuclear!, normNuclear, λ, kargs)
  elseif name=="Positive"
    return Regularization(proxPositive!, normPositive, λ, kargs)
  elseif name=="Proj"
    return Regularization(proxProj!, normProj, λ, kargs)
  else
    error("Regularization $name not found.")
  end

  return Regularization(proxL2!, normL2, 0.0, kargs)
end

"""
    Regularization(names::Vector{String}, λ::Vector{Float64}; kargs...)

create a Regularization object containing all the infos necessary to calculate a proximal map.
Valid names are the same as in `Regularization(name::String, λ::AbstractFloat; kargs...)`.
"""
function Regularization(names::Vector{String},
                        λ::Vector{Float64}; kargs...)
  #@Mirco I do not know what to do with kargs here
  if length(names) != length(λ)
    @error names " and " λ " need to have the same length "
  end
  reg = Regularization[]
  for l=1:length(names)
    push!(reg, Regularization(names[l],λ[l]; kargs...))
  end
  return reg
end

"""
    normalize!(reg::Regularization, data)

scales the regularization parameter depending of the energy of the data (in-place).
"""
function normalize!(reg::Regularization, data)
  meanEnergy = norm(data,1)/length(data)
  reg.λ = meanEnergy*reg.λ
end

norm0(x::Array{T}, λ::Float64; sparseTrafo::Trafo=nothing, kargs...) where T = 0.0
