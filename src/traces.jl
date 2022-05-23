export Trace, Traces, MultiplexTraces

import MacroTools: @forward

#####

struct Trace{T,E} <: AbstractVector{E}
    parent::T
end

function Trace(x::T) where {T<:AbstractArray}
    E = eltype(x)
    N = ndims(x) - 1
    P = typeof(x)
    I = Tuple{ntuple(_ -> Base.Slice{Base.OneTo{Int}}, Val(ndims(x) - 1))...,Int}
    Trace{T,SubArray{E,N,P,I,true}}(x)
end

Base.convert(::Type{Trace}, x::AbstractArray) = Trace(x)

Base.size(x::Trace) = (size(x.parent, ndims(x.parent)),)
Base.getindex(s::Trace, I) = Base.maybeview(s.parent, ntuple(i -> i == ndims(s.parent) ? I : (:), Val(ndims(s.parent)))...)
Base.setindex!(s::Trace, v, I) = setindex!(s.parent, v, ntuple(i -> i == ndims(s.parent) ? I : (:), Val(ndims(s.parent)))...)

@forward Trace.parent Base.parent, Base.pushfirst!, Base.push!, Base.append!, Base.prepend!, Base.pop!, Base.popfirst!, Base.empty!

#####

"""
For each concrete `AbstractTraces`, we have the following assumption:

1. Every inner trace is an `AbstractVector`
1. Support partial updating
1. Return *View* by default when getting elements.
"""
abstract type AbstractTraces{names,T} <: AbstractVector{NamedTuple{names,T}} end

Base.keys(t::AbstractTraces{names}) where {names} = names
Base.haskey(t::AbstractTraces{names}, k::Symbol) where {names} = k in names

#####

"""
    Traces(;kw...)
"""
struct Traces{T,names,E} <: AbstractTraces{names,E}
    traces::T
    function Traces(; kw...)
        data = map(x -> convert(Trace, x), values(kw))
        new{typeof(data),keys(data),Tuple{typeof(data).types...}}(data)
    end
end

Base.getindex(t::Traces, s::Symbol) = getindex(t.traces, s)
Base.getindex(t::Traces, i) = map(x -> getindex(x, i), t.traces)

@forward Traces.traces Base.parent

Base.size(t::Traces) = (mapreduce(length, min, t.traces),)

for f in (:push!, :pushfirst!, :append!, :prepend!)
    @eval function Base.$f(ts::Traces, xs::NamedTuple)
        for (k, v) in pairs(xs)
            $f(ts.traces[k], v)
        end
    end
end

for f in (:pop!, :popfirst!, :empty!)
    @eval Base.$f(ts::Traces) = map($f, ts.traces)
end

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
struct MultiplexTraces{names,T,E} <: AbstractTraces{names,Tuple{E,E}}
    trace::T
end

function MultiplexTraces{names}(t::AbstractVector) where {names}
    if length(names) != 2
        throw(ArgumentError("MultiplexTraces has exactly two sub traces, got $length(names) trace names"))
    end
    trace = convert(Trace, t)
    MultiplexTraces{names,typeof(trace),eltype(trace)}(trace)
end

function Base.getindex(t::MultiplexTraces{names}, k::Symbol) where {names}
    a, b = names
    if k == a
        Trace(t.trace[1:end-1])
    elseif k == b
        Trace(t.trace[2:end])
    else
        throw(ArgumentError("unknown trace name: $k"))
    end
end

Base.getindex(t::MultiplexTraces{names}, I::Int) where {names} = NamedTuple{names}((t.trace[I], t.trace[I+1]))
Base.getindex(t::MultiplexTraces{names}, I::AbstractArray{Int}) where {names} = NamedTuple{names}((t.trace[I], t.trace[I.+1]))

Base.size(t::MultiplexTraces) = (max(0, length(t.trace) - 1),)

@forward MultiplexTraces.trace Base.parent, Base.pop!, Base.popfirst!, Base.empty!

for f in (:push!, :pushfirst!, :append!, :prepend!)
    @eval function Base.$f(t::MultiplexTraces{names}, x::NamedTuple{ks,Tuple{Ts}}) where {names,ks,Ts}
        k, v = first(ks), first(x)
        if k in names
            $f(t.trace, v)
        else
            throw(ArgumentError("unknown trace name: $k"))
        end
    end
end

#####

struct MergedTraces{names,T,N,E} <: AbstractTraces{names,E}
    traces::T
    inds::NamedTuple{names,NTuple{N,Int}}
end

Base.getindex(ts::MergedTraces, s::Symbol) = ts.traces[ts.inds[s]][s]

function Base.:(+)(t1::AbstractTraces{k1,T1}, t2::AbstractTraces{k2,T2}) where {k1,k2,T1,T2}
    ks = (k1..., k2...)
    ts = (t1, t2)
    inds = (; (k => 1 for k in k1)..., (k => 2 for k in k2)...)
    MergedTraces{ks,typeof(ts),length(ks),Tuple{T1.types...,T2.types...}}(ts, inds)
end

function Base.:(+)(t1::AbstractTraces{k1,T1}, t2::MergedTraces{k2,T,N,T2}) where {k1,T1,k2,T,N,T2}
    ks = (k1..., k2...)
    ts = (t1, t2.traces...)
    inds = merge(NamedTuple(k => 1 for k in k1), map(v => v + 1, t1.inds))
    MergedTraces{ks,typeof(ts),length(ks),Tuple{T1.types...,T2.types...}}(ts, inds)
end


function Base.:(+)(t1::MergedTraces{k1,T,N,T1}, t2::AbstractTraces{k2,T2}) where {k1,T,N,T1,k2,T2}
    ks = (k1..., k2...)
    ts = (t1.traces..., t2)
    inds = merge(t1.inds, (; (k => length(ts) for k in k2)...))
    MergedTraces{ks,typeof(ts),length(ks),Tuple{T1.types...,T2.types...}}(ts, inds)
end

function Base.:(+)(t1::MergedTraces{k1,T1,N1,E1}, t2::MergedTraces{k2,T2,N2,E2}) where {k1,T1,N1,E1,k2,T2,N2,E2}
    ks = (k1..., k2...)
    ts = (t1.traces..., t2.traces...)
    inds = merge(t1.inds, map(x -> x + length(t1.traces), t2.inds))
    MergedTraces{ks,typeof(ts),length(ks),Tuple{T1.types...,T2.types...}}(ts, inds)
end


Base.size(t::MergedTraces) = (mapreduce(length, min, t.traces),)
Base.getindex(t::MergedTraces, I) = mapreduce(x -> getindex(x, I), merge, t.traces)

for f in (:push!, :pushfirst!, :append!, :prepend!)
    @eval function Base.$f(ts::MergedTraces, xs::NamedTuple)
        for (k, v) in pairs(xs)
            $f(ts.traces[ts.inds[k]], (; k => v))
        end
    end
end

for f in (:pop!, :popfirst!, :empty!)
    @eval function Base.$f(ts::MergedTraces)
        for t in ts.traces
            $f(t)
        end
    end
end
