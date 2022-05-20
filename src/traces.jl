export Trace, Traces, sample

import MacroTools: @forward

"""
    AbstractTrace{T} <: AbstractVector{T}

An `AbstractTrace` is a subtype of `AbstractVector`. The following methods should be implemented:

- `Base.length`
- `Base.firstindex`
- `Base.lastindex`
- `Base.getindex`
- `Base.view`
- `Base.push!`
- `Base.append!`
- `Base.empty!`
- `Base.pop!`
- `Base.popfirst!`
"""
abstract type AbstractTrace{T} <: AbstractVector{T} end

"""
    AbstractTraces{names}

An `AbstractTraces` is a group of different [`AbstractTrace`](@ref). Following methods must be implemented:

- `Base.getindex`, get the inner `AbstractTrace` given a trace name.
- `Base.keys`
- `Base.haskey`
- `Base.push!`
- `Base.append!`
- `Base.pop!`
- `Base.popfirst!`
- `Base.empty!`
"""
abstract type AbstractTraces{names} end

Base.keys(t::AbstractTraces{names}) where {names} = names
Base.haskey(t::AbstractTraces{names}) where {names} = haskey(names)

Base.push!(t::AbstractTraces; kw...) = push!(t, values(kw))

function Base.push!(t::AbstractTraces, x::NamedTuple)
    for k in keys(x)
        push!(t[k], x[k])
    end
end

Base.append!(t::AbstractTraces; kw...) = append!(t, values(kw))

function Base.append!(t::AbstractTraces, x::NamedTuple)
    for k in keys(x)
        append!(t[k], x[k])
    end
end


#####

"""
    Trace(data)

The most common [`AbstractTrace`](@ref). A wrapper of arbitrary container.
Generally we assume the `data` is an `AbstractVector` like object. When an
`AbstractArray` is given, we view it as a vector of sub-arrays along the last
dimension.
"""
struct Trace{T} <: AbstractTrace
    x::T
end

@forward Trace.x Base.length, Base.lastindex, Base.firstindex, Base.getindex, Base.view, Base.push!, Base.append!, Base.pop!, Base.popfirst!, Base.empty!

Base.convert(::Type{Trace}, x) = Trace(x)

Base.length(t::Trace{<:AbstractArray}) = size(t.x, ndims(t.x))
Base.getindex(t::Trace{<:AbstractArray}, I...) = getindex(t.x, ntuple(_ -> :, ndims(t.x) - 1)..., I...)
Base.view(t::Trace{<:AbstractArray}, I...) = view(t.x, ntuple(_ -> :, ndims(t.x) - 1)..., I...)

#####

"""
    Traces(;kw...)

A container of several named-[`AbstractTrace`](@ref)s. Each element in the `kw` will be converted into a `Trace`.
"""
struct Traces{names,T} <: AbstractTraces
    traces::NamedTuple{names,T}
    function Traces(; kw...)
        traces = map(x -> convert(Trace, x), values(kw))
        new{keys(traces),typeof(values(traces))}(traces)
    end
end

@forward Traces.traces Base.getindex

Base.pop!(t::Traces) = map(pop!, t.traces)
Base.popfirst!(t::Traces) = map(popfirst!, t.traces)
Base.empty!(t::Traces) = map(empty!, t.traces)

#####

"""
    MultiplexTraces{names}(trace)

A special [`AbstractTraces`](@ref) which has exactly two traces of the same
length. And those two traces share the header and tail part.

For example, if a `trace` contains elements between 0 and 9, then the first
`trace_A` is a view of elements from 0 to 8 and the second one is a view from 1
to 9.

```
      ┌─────trace_A───┐
trace 0 1 2 3 4 5 6 7 8 9
        └────trace_B────┘
```

This is quite common in RL to represent `states` and `next_states`.
"""
struct MultiplexTraces{names,T} <: AbstractTraces{names}
    trace::Trace{T}
end

function MultiplexTraces{names}(trace) where {names}
    if length(names) != 2
        throw(ArgumentError("MultiplexTraces has exactly two sub traces, got $length(names) trace names"))
    end
    t = convert(Trace, trace)
    MultiplexTraces{names,typeof(t)}(t)
end

@forward MultiplexTraces.trace Base.pop!, Base.popfirst!, Base.empty!

Base.getindex(t::MultiplexTraces, i::Int) = getindex(t, keys(t)[i])

function Base.getindex(t::MultiplexTraces, k::Symbol)
    a, b = keys(t)
    if k == a
        @view t.trace[1:end-1]
    elseif k == b
        @view t.trace[2:end]
    else
        throw(ArgumentError("unknown trace name: $k"))
    end
end

function Base.push!(t::MultiplexTraces, x::NamedTuple{ks,Tuple{Ts}}) where {ks,Ts}
    k, v = first(ks), first(x)
    if k in keys(t)
        push!(t.trace, v)
    else
        throw(ArgumentError("unknown trace name: $k"))
    end
end

function Base.append!(t::MultiplexTraces, x::NamedTuple{ks,Tuple{Ts}}) where {ks,Ts}
    k, v = first(ks), first(x)
    if k in keys(t)
        append!(t.trace, v)
    else
        throw(ArgumentError("unknown trace name: $k"))
    end
end

#####

struct MergedTraces{names,T,N} <: AbstractTraces{names}
    traces::T
    inds::NamedTuple{names,NTuple{N,Int}}
end

function Base.(:*)(t1::AbstractTraces, t2::AbstractTraces)
    k1, k2 = keys(t1), keys(t2)
    ks = (k1..., k2...)
    ts = (t1, t2)
    inds = (; (k => 1 for k in k1)..., (k => 2 for k in k2)...)
    MergedTraces{ks,typeof(ts)}(ts, inds)
end

function Base.(:*)(t1::AbstractTraces, t2::MergedTraces)
    k1, k2 = keys(t1), keys(t2)
    ks = (k1..., k2...)
    ts = (t1, t2.traces...)
    inds = (; (k => 1 for k in k1)..., map(x -> x + 1, t2.inds)...)
    MergedTraces{ks,typeof(ts)}(ts, inds)
end

function Base.(:*)(t1::MergedTraces, t2::AbstractTraces)
    k1, k2 = keys(t1), keys(t2)
    ks = (k1..., k2...)
    ts = (t1.traces..., t2)
    inds = merge(t1.inds, (; (k => length(t1.traces) + 1 for k in k2)...))
    MergedTraces{ks,typeof(ts)}(ts, inds)
end

function Base.push!(ts::MergedTraces, xs::NamedTuple)
    for (k, v) in pairs(xs)
        t = ts.traces[t.inds[k]]
        push!(t, v)
    end
end

function Base.append!(ts::MergedTraces, xs::NamedTuple)
    for (k, v) in pairs(xs)
        t = ts.traces[t.inds[k]]
        append!(t, v)
    end
end

function Base.pop!(ts::MergedTraces)
    for t in ts.traces
        pop!(t)
    end
end

function Base.popfirst!(ts::MergedTraces)
    for t in ts.traces
        popfirst!(t)
    end
end

function Base.empty!(ts::MergedTraces)
    for t in ts.traces
        empty!(t)
    end
end