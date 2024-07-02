function RegularizedLeastSquares.iterate_row_index(solver::Kaczmarz, state::RegularizedLeastSquares.KaczmarzState{T, vecT}, A, row, index) where {T, vecT <: AbstractGPUArray}
  state.τl = RegularizedLeastSquares.dot_with_matrix_row(A,state.x,row)
  @allowscalar state.αl = solver.denom[index]*(state.u[row]-state.τl-state.ɛw*state.vl[row])
  RegularizedLeastSquares.kaczmarz_update!(A,state.x,row,state.αl)
  @allowscalar state.vl[row] += state.αl*state.ɛw
end

function RegularizedLeastSquares.kaczmarz_update!(A, x::vecT, row, beta) where {T, vecT <: AbstractGPUVector{T}}
  x[:] .=  x .+ beta * conj.(view(A, row, :))
end