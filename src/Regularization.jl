export Regularization

type Regularization
  L2::Bool
  L1::Bool
  L21::Bool
  TV::Bool
  LLR::Bool
  Positive::Bool
  params
end

"""
Return a list of all available Regularizations
"""
function RegularizationList()
  Any["L2", "L1", "L21", "TV", "LLR", "Positive"]
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
  params[:slices] = 1
  params[:blockSize] = [1,1]
  return params
end

#
# params: lambda, alpha, beta, gamma, ...
# blockSize -> LLR
# directions / directionWeights -> TV
#
function Regularization(;L1=false,L2=false,L21=false,TV=false,LLR=false,Positive=false, kargs...)
  params = merge(regParamsDefault(), Dict(kargs))
  return Regularization(L2,L1,L21,TV,LLR,Positive,params)
end


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

end
