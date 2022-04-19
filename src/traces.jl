export Trace, Traces, Episode, Episodes, pop!!, popfirst!!

"""
    Trace(data)

A wrapper of arbitrary container. Generally we assume the `data` is an
`AbstractVector` like object. When an `AbstractArray` is given, we view it as a
vector of sub-arrays along the last dimension.
"""
struct Trace{T}
    x::T
end

Base.length(t::Trace) = length(t.x)
Base.lastindex(t::Trace) = length(t)
Base.firstindex(t::Trace) = 1
Base.length(t::Trace{<:AbstractArray}) = size(t.x, ndims(t.x))

Base.convert(::Type{Trace}, x) = Trace(x)

Base.getindex(t::Trace{<:AbstractVector}, I...) = getindex(t.x, I...)
Base.view(t::Trace{<:AbstractVector}, I...) = view(t.x, I...)

Base.getindex(t::Trace{<:AbstractArray}, I...) = getindex(t.x, ntuple(_ -> :, ndims(t.x) - 1)..., I...)
Base.view(t::Trace{<:AbstractArray}, I...) = view(t.x, ntuple(_ -> :, ndims(t.x) - 1)..., I...)

Base.push!(t::Trace, x) = push!(t.x, x)
Base.append!(t::Trace, x) = append!(t.x, x)

Base.pop!(t::Trace) = pop!(t.x)
Base.popfirst!(t::Trace) = popfirst!(t.x)
Base.empty!(t::Trace) = empty!(t.x)

#####

```
    Traces(;kw...)

A container of several named-[`Trace`](@ref)s. Each element in the `kw` will be converted into a `Trace`.
```
struct Traces{names,T}
    traces::NamedTuple{names,T}
    function Traces(; kw...)
        traces = map(x -> convert(Trace, x), values(kw))
        new{keys(traces),typeof(values(traces))}(traces)
    end
end

Base.keys(t::Traces) = keys(t.traces)
Base.haskey(t::Traces, s::Symbol) = haskey(t.traces, s)
Base.getindex(t::Traces, x) = getindex(t.traces, x)

Base.push!(t::Traces; kw...) = push!(t, values(kw))

function Base.push!(t::Traces, x::NamedTuple)
    for k in keys(x)
        push!(t[k], x[k])
    end
end

Base.append!(t::Traces; kw...) = append!(t, values(kw))

function Base.append!(t::Traces, x::NamedTuple)
    for k in keys(x)
        append!(t[k], x[k])
    end
end

Base.pop!(t::Traces) = map(pop!, t.traces)
Base.popfirst!(t::Traces) = map(popfirst!, t.traces)
Base.empty!(t::Traces) = map(empty!, t.traces)

#####

struct Episode{T}
    traces::T
    is_done::Ref{Bool}
end

Base.getindex(e::Episode) = getindex(e.is_done)
Base.setindex!(e::Episode, x::Bool) = setindex!(e.is_done, x)

Episode(t::Traces) = Episode(t, Ref(false))

function Base.append!(t::Episode, x)
    if t.is_done[]
        throw(ArgumentError("The episode is already flagged as done!"))
    else
        append!(t.traces, x)
    end
end

function Base.push!(t::Episode, x)
    if t.is_done[]
        throw(ArgumentError("The episode is already flagged as done!"))
    else
        push!(t.traces, x)
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

struct Episodes{T}
    episodes::Vector{Episode{T}}
    lengths::Vector{Int}  # TODO: sum tree?
    total_length::Ref{Int}
end

function Base.push!(e::Episodes, x)
    push!(e.episodes[end], x)
    diff = length(e.episodes[end]) - e.lengths[end]
    e.lengths[end] += diff
    e.total_length[] += diff
end

function Base.push!(e::Episodes, x::Episode)
    push!(e.episodes, x)
    push!(e.lengths, length(x))
    e.total_length[] += length(x)
end

function Base.append!(e::Episodes, x)
    append!(e.episodes[end], x)
    diff = length(e.episodes[end]) - e.lengths[end]
    e.lengths[end] += diff
    e.total_length[] += diff
end

function Base.append!(e::Episodes, x::AbstractVector{<:Episode})
    append!(e.episodes, x)
    push!(e.lengths, length(x))
    e.total_length[] += diff
end

function Base.pop!(e::Episodes)
    e.total_length[] -= e.lengths[end]
    pop!(e.episodes)
    pop!(e.lengths)
end

function pop!!(e::Episodes)
    pop!(e.episodes[end])
    diff = e.lengths[end] - length(e.episodes[end])
    e.lengths[end] = length(e.episodes[end])
    e.total_length[] -= diff
end

function Base.popfirst!(e::Episodes)
    diff = length(e.episodes[begin])
    e.total_length[] -= diff
    popfirst!(e.episodes)
    popfirst!(e.lengths)
end

function popfirst!!(e::Episodes)
    popfirst!(e.episodes)
    diff = e.lengths[begin] - length(e.episodes[begin])
    e.lengths[begin] = length(e.episodes[begin])
    e.total_length[] -= diff
end