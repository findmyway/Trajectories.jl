@testset "trajectories" begin
    t = Trajectory(
        traces=Traces(
            a=Int[],
            b=Bool[]
        ),
        sampler=BatchSampler(3)
    )

    for i in 1:10
        push!(t; a=i, b=isodd(i))
    end

    batch = rand(t)

    @test size(batch[:a]) == (3,)
    @test size(batch[:b]) == (3,)
end