using Test
using Trajectories
import Trajectories.normalize
import OnlineStats: fit!, mean, std

@testset "normalization.jl" begin
    #scalar normalization
    rewards = [1.:10;]
    rn = scalar_normalizer()
    fit!(rn, rewards)
    batch_reward = normalize(rn, [6.,5.,10.])
    @test batch_reward ≈ ([6.,5.,10.] .- mean(1:10))./std(1:10)
    #vector normalization
    states = reshape([1:50;], 5, 10)
    sn = array_normalizer((5,))
    fit!(sn, states)
    @test [mean(stat) for stat in sn] == [mean((1:5:46) .+i) for i in 0:4]
    batch_states =normalize(sn, reshape(repeat(5.:-1:1, 5), 5,5))
    @test all(length(unique(x)) == 1 for x in eachrow(batch_states))
    #array normalization
    states = reshape(1.:250, 5,5,10)
    sn = array_normalizer((5,5))
    fit!(sn, states)
    batch_states = normalize(sn, collect(states))
    
    #NormalizedTrace
    t = Trajectory(
        container=Traces(
            a= NormalizedTrace(Float32[], scalar_normalizer()),
            b=Int[],
            c=NormalizedTrace(Vector{Float32}[], array_normalizer((10,))) #TODO check with ElasticArrays and Episodes
        ),
        sampler=BatchSampler(300000),
        controler=InsertSampleRatioControler(Inf, 0)
    )
    append!(t, a = [1,2,3], b = [1,2,3], c = eachcol(reshape(1f0:30, 10,3)))
    push!(t, a = 2, b = 2, c = fill(mean(1:30), 10))
    @test mean(t.container[:a].trace.x) ≈ 2.
    @test std(t.container[:a].trace.x) ≈ std([1,2,2,3])
    a,b,c = take!(t)
    @test eltype(a) == Float32
    @test mean(a) ≈ 0 atol = 0.01
    @test mean(b) ≈ 2 atol = 0.01 #b is not normalized
    @test eltype(first(c)) == Float32
    @test all(isapprox(0f0, atol = 0.01), vec(mean(reduce(hcat,c), dims = 2)))
end