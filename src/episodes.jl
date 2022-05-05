export Episode, Episodes

"""
    Episode(traces)

An `Episode` is a wrapper around [`Traces`](@ref). You can use `(e::Episode)[]`
to check/update whether the episode reaches a terminal or not.
"""
struct Episode{T}
    traces::T
    is_done::Ref{Bool}
end

Base.getindex(e::Episode, s::Symbol) = getindex(e.traces, s)
Base.keys(e::Episode) = keys(e.traces)

Base.getindex(e::Episode) = getindex(e.is_done)
Base.setindex!(e::Episode, x::Bool) = setindex!(e.is_done, x)

Base.length(e::Episode) = length(e.traces)

Episode(t::Traces) = Episode(t, Ref(false))

function Base.push!(t::Episode, x)
    if t.is_done[]
        throw(ArgumentError("The episode is already flagged as done!"))
    else
        push!(t.traces, x)
    end
end

function Base.append!(t::Episode, x)
    if t.is_done[]
        throw(ArgumentError("The episode is already flagged as done!"))
    else
        append!(t.traces, x)
    end
end

function Base.pop!(t::Episode)
    pop!(t.traces)
    t.is_done[] = false
end

Base.popfirst!(t::Episode) = popfirst!(t.traces)

function Base.empty!(t::Episode)
    empty!(t.traces)
    t.is_done[] = false
end

#####

"""
    Episodes(init)

A container for multiple [`Episode`](@ref)s. `init` is a parameterness function which return an [`Episode`](@ref).
"""
struct Episodes
    init::Any
    episodes::Vector{Episode}
    inds::Vector{Tuple{Int,Int}}
end

Base.length(e::Episodes) = length(e.inds)

function Base.push!(e::Episodes, x::Episode)
    push!(e.episodes, x)
    for i in 1:length(x)
        push!(e.inds, (length(e.episodes), i))
    end
end

function Base.append!(e::Episodes, xs::AbstractVector{<:Episode})
    for x in xs
        push!(e, x)
    end
end

function Base.push!(e::Episodes, x)
    if isempty(e.episodes) || e.episodes[end][]
        episode = e.init()
        push!(episode, x)
        push!(e.episodes, episode)
    else
        push!(e.episodes[end], x)
        push!(e.inds, (length(e.episodes), length(e.episodes[end])))
    end
end

function Base.append!(e::Episodes, x)
    n_pre = length(e.episodes[end])
    append!(e.episodes[end], x)
    n_post = length(e.episodes[end])
    for i in n_pre:n_post
        push!(e.inds, (lengthe.episodes, i))
    end
end

##

function sample(s::BatchSampler, e::Episodes)
    inds = rand(s.rng, 1:length(t), s.batch_size)
    # TODO: batch
end