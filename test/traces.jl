@testset "Trace 1d" begin
    t = Trace([])
    @test length(t) == 0

    push!(t, 1)
    @test length(t) == 1
    @test t[1] == 1

    append!(t, [2, 3])
    @test length(t) == 3
    @test @view(t[2:3]) == [2, 3]

    pop!(t)
    @test length(t) == 2

    s = BatchSampler(2)
    @test size(sample(s, t)) == (2,)

    empty!(t)
    @test length(t) == 0

end

@testset "Trace 2d" begin
    t = Trace([
        1 2 3
        4 5 6
    ])
    @test length(t) == 3
    @test t[1] == [1, 4]
    @test @view(t[2:3]) == [2 3; 5 6]

    s = BatchSampler(5)
    @test size(sample(s, t)) == (2, 5)
end

@testset "Traces" begin
    t = Traces(;
        a=[1, 2],
        b=Bool[0, 1]
    )

    @test keys(t) == (:a, :b)
    @test haskey(t, :a)
    @test t[:a] isa Trace

    push!(t; a=3, b=true)
    @test t[:a][end] == 3
    @test t[:b][end] == true

    append!(t; a=[4, 5], b=[false, false])
    @test length(t[:a]) == 5
    @test t[:b][end-1:end] == [false, false]

    s = BatchSampler(5)
    @test size(sample(s, t)[:a]) == (5,)
end