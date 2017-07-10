export proxPositive!

@doc "enforce positivity and realness of solution." ->
function proxPositive!(reg::Regularization, x)
  proxPositive!(x)
end

function proxPositive!(x)
  enfReal!(x)
  enfPos!(x)
end
