struct Trajectory{T,S}
    traces::T
    sampler::S
end

Base.rand(t::Trajectory) = rand(t.sampler, t.traces)

Base.push!(t::Trajectory; kw...) = push!(t, values(kw))
Base.push!(t::Trajectory, x) = push!(t.traces, x)

Base.append!(t::Trajectory; kw...) = append!(t, values(kw))
Base.append!(t::Trajectory, x) = append(t.traces, x)

Base.pop!(t::Trajectory) = pop!(t.traces)
Base.empty!(t::Trajectory) = empty!(t.traces)

Base.getindex(t::Trajectory, k) = getindex(t.traces, k)
Base.length(t::Trajectory) = length(t.traces)
