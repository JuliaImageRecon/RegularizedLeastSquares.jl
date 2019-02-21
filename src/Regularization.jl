export Regularization, getRegularization, lambdList, prox! #, norm

abstract type AbstractRegularization end

#
# each Regularization should contain:
#   1. prox!(x,λ;kargs...) : function to evaluate the proximal mapping (in-place in x)
#   2. norm(x,λ;kargs...) : function to evaluate the Regularization norm
#   3. λ :         Regularization parameter
#
mutable struct Regularization <: AbstractRegularization
  prox!::Function
  norm::Function
  λ::Float64
  params::Dict{Symbol,Any}
end

Regularization(;prox = x->x, norm = x->0.0, λ=0.0) = Regularization(prox!,norm,λ,Dict{Symbol,Any}())

"""
Return a list of all available Regularizations
"""
function RegularizationList()
  Any["L2", "L1", "L21", "TV", "LLR", "Positive", "Proj", "Nuclear"]
end

"""
 create a Regularization object containing all the infos necessary to calculate a proximal map
"""
function getRegularization(name::String, λ::Float; kargs...)
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

###################
# utility functions
###################
function normalize!(reg::Regularization, data)
  meanEnergy = norm(data,1)/length(data)
  reg.λ = meanEnergy*reg.λ
end
