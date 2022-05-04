@testset "trajectories" begin
    t = Trajectory(
        container=Traces(
            a=Int[],
            b=Bool[]
        ),
        sampler=BatchSampler(3),
        controler=InsertSampleRatioControler(0.25, 4)
    )

    batches = []

    for batch in t
        push!(batches, batch)
    end

    @test length(batches) == 0  # threshold not reached yet

    append!(t; a=[1, 2, 3], b=[false, true, false])

    for batch in t
        push!(batches, batch)
    end

    @test length(batches) == 0  # threshold not reached yet

    push!(t; a=4, b=true)

    for batch in t
        push!(batches, batch)
    end

    @test length(batches) == 1  # 4 inserted, ratio is 0.25

    append!(t; a=[5, 6, 7], b=[true, true, true])

    for batch in t
        push!(batches, batch)
    end

    @test length(batches) == 2  # 7 inserted, ratio is 0.25

    push!(t; a=8, b=true)

    for batch in t
        push!(batches, batch)
    end

    @test length(batches) == 2  # 8 inserted, ratio is 0.25

    n = 100
    for i in 1:n
        append!(t; a=[i, i, i, i], b=[false, true, false, true])
    end

    s = 0
    for _ in t
        s += 1
    end
    @test s == n
end