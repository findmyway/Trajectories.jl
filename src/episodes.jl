export Episode, Episodes


"""
    Episode(traces)

An `Episode` is a wrapper around [`Traces`](@ref). You can use `(e::Episode)[]`
to check/update whether the episode reaches a terminal or not.
"""
struct Episode{T,E} <: AbstractVector{E}
    traces::T
    is_terminated::Ref{Bool}
end

Base.getindex(e::Episode, I) = getindex(e.traces, I)
Base.getindex(e::Episode) = getindex(e.is_done)
Base.setindex!(e::Episode, x::Bool) = setindex!(e.is_done, x)

Base.size(e::Episode) = size(e.traces)

Episode(t::T) where {T<:AbstractTraces} = Episode{T,eltype(t)}(t, Ref(false))

for f in (:push!, :pushfirst!, :append!, :prepend!)
    @eval function Base.$f(t::Episode, x)
        if t.is_done[]
            throw(ArgumentError("The episode is already flagged as done!"))
        else
            $f(t.traces, x)
        end
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
struct Episodes <: AbstractVector{Episode}
    init::Any
    episodes::Vector{Episode}
    inds::Vector{Tuple{Int,Int}}
end

Base.size(e::Episodes) = size(e.inds)
Base.getindex(e::Episodes, I) = getindex(e.episodes, I)

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
