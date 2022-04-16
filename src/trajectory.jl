struct Trajectory{T,S}
    traces::T
    sampler::S
end

Base.rand(t::Trajectory) = rand(t.sampler, t.traces)
Base.push!(t::Trajectory, x) = push!(t.traces, x)
Base.append!(t::Trajectory, x) = append(t.traces, x)
