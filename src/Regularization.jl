import Base.A_mul_B!, Base.norm
export Regularization, getRegularization, lambdList, prox! #, norm

type Regularization
  L2::Bool
  L1::Bool
  L21::Bool
  TV::Bool
  LLR::Bool
  Positive::Bool
  Nuclear::Bool
  params
end

"""
Return a list of all available Regularizations
"""
function RegularizationList()
  Any["L2", "L1", "L21", "TV", "LLR", "Positive", "Nuclear"]
end

"""
Return a list of all parameters determining regularization strength
"""
function lambdList()
  Any["lambdL2", "lambdL1", "lambdL21", "lambdTV", "lambdLLR", "lambdNuclear"]
end

"""
 Return a default set of parameters for the regularizers
"""
function regParamsDefault()
  params = Dict{Symbol,Any}()
  params[:lambdL2] = 0.
  params[:lambdL1] = 0.
  params[:lambdL21] = 0.
  params[:lambdTV] = 0.
  params[:lambdLLR] = 0.
  params[:lambdNuclear] = 0.
  params[:slices] = 1
  params[:blockSize] = [1,1]
  return params
end

"""
 create a Regularization object containing all the infos necessary to calculate a proximal map
"""
function Regularization(;L1=false,L2=false,L21=false,TV=false,LLR=false,Positive=false,Nuclear=false, kargs...)
  params = merge(regParamsDefault(), Dict(kargs))
  return Regularization(L2,L1,L21,TV,LLR,Positive,Nuclear,params)
end

function getRegularization(name::String; kargs...)
  if name=="L2"
    return Regularization(;L2=true, kargs...)
  elseif name=="L1"
    return Regularization(;L1=true, kargs...)
  elseif name=="L21"
    return Regularization(;L21=true, kargs...)
  elseif name=="TV"
    return Regularization(;TV=true, kargs...)
  elseif name=="LLR"
    return Regularization(;LLR=true, kargs...)
  elseif name=="Nuclear"
    return Regularization(;Nuclear=true, kargs...)
  elseif name=="Positive"
    return Regularization(;Positive=true, kargs...)
  else
    error("Regularization $name not found.")
  end

  return Regularization()
end

function getRegularization(names::Vector{String}; kargs...)
  params = Dict(kargs)
  for i=RegularizationList()
    contains(==,names,i) ? params[Symbol(i)]=true : continue
  end

  return Regularization(;params...)
end

"""
calculate proximal map
"""
function prox!(reg::Regularization, x)

  # prepend dedicated proximal maps for combined regularizations and end with break
  # otherwise approximate the combined proximal map as a composition of individual maps
  if reg.L2
    proxL2!(reg,x)
  end
  if reg.L1
    proxL1!(reg,x)
  end
  if reg.L21
    proxL21!(reg,x)
  end
  if reg.TV
    proxTV!(reg,x)
  end
  if reg.LLR
    proxLLR!(reg,x)
  end
  if reg.Positive
    proxPositive!(reg,x)
  end
  if reg.Nuclear
    proxNuclear!(reg,x)
  end

end

###################
# utility functions
###################
function A_mul_B!(reg::Regularization, x::Real)
  for lambd in lambdList()
    reg.params[Symbol(lambd)] *= x
  end
end

function norm(reg::Regularization,x)
  res = 0.

  if reg.L2
    res += normL2(reg,x)
  end
  if reg.L1
    res += normL1(reg,x)
  end
  if reg.L21
    res += normL21(reg,x)
  end
  if reg.TV
    res += normTV(reg,x)
  end
  if reg.LLR
    res += normLLR(reg,x)
  end

  return res
end
