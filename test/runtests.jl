using Trajectories
using Test

@testset "Trajectories.jl" begin
    include("traces.jl")
    include("trajectories.jl")
    include("normalization.jl")
end
