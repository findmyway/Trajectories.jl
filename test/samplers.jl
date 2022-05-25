@testset "BatchSampler" begin
    sz = 32
    s = BatchSampler(sz)
    t = Traces(
        state=rand(3, 4, 5),
        action=rand(1:4, 5),
    )

    b = Trajectories.sample(s, t)

    @test keys(b) == (:state, :action)
    @test size(b.state) == (3, 4, sz)
    @test size(b.action) == (sz,)

    e = Episodes() do
        Episode(Traces(state=rand(2, 3, 0), action=rand(0)))
    end

    push!(e, Episode(Traces(state=rand(2, 3, 2), action=rand(2))))
    push!(e, Episode(Traces(state=rand(2, 3, 3), action=rand(3))))

    @test length(e) == 5
    @test size(e[2:4].state) == (2, 3, 3)
    @test_broken size(e[2:4].action) == (3,)
end