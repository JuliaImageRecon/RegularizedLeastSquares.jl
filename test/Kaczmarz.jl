@testset "test Kaczmarz update" begin
    for T in [Float32,Float64,ComplexF32,ComplexF64]
        # set up
	M = 127
	N = 16
	
	A = rand(T,M,N)
	Aᵀ = transpose(A)
	b = zeros(T,M)
	β = rand(T)
	k = rand(1:N)
	# end set up
	
	RegularizedLeastSquares.kaczmarz_update!(Aᵀ,b,k,β)
	@test b ≈ β*conj(A[:,k])
    end

    for T in [Float32,Float64,ComplexF32,ComplexF64]
        # set up
        M = 127
        N = 16

        A = rand(T,N,M)
        b = zeros(T,M)
        β = rand(T)
        k = rand(1:N)
        # end set up

	RegularizedLeastSquares.kaczmarz_update!(A,b,k,β)
        @test b ≈ β*conj(A[k,:])
    end
end
