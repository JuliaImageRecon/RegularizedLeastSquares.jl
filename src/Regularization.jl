export Regularization

immutable Regularization
  L2::Bool
  L1::Bool
  TV::Bool
  LLR::Bool
  Positive::Bool
  params
end

#
# params: lambda, alpha, beta, gamma, ...
# blockSize -> LLR
# directions / directionWeights -> TV
#

function Regularization(;L1=false,L2=false,TV=false,LLR=false,Positive=false, kargs...)

  # error() if combinations do not make sense

  return Regularization(L2,L1,TV,LLR,Positive,Dict(kargs))
end


function prox(reg::Regularization, x)
  if reg.L2
    println("Hallo")
    return proxL2(reg,x)
  elseif reg.L1
    return proxL1(reg,x)
  elseif reg.TV

  elseif reg.LLR
    return proxLLR(reg,x)
  else
    println("fallback")
  end

end


function proxL2(reg,x)

end

function proxL1(reg,x)

end

function proxLLR(reg,x)

end
