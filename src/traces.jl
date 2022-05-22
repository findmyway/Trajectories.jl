export Trace, Traces, MultiplexTraces

using StructArrays
import MacroTools: @forward

abstract type AbstractTraces{names,T} <: AbstractVector{NamedTuple{names,T}} end

# function Base.show(io::IO, ::MIME"text/plain", t::AbstractTraces{names}) where {names}
#     println(io, "$(length(names)) traces in total with $(length(t)) elements:")
#     for n in names
#         println("  :$n => $(summary(t[n]))")
#     end
# end

"""
    Traces(;kw...)
"""
struct Traces{T,names,E} <: AbstractTraces{names,E}
    traces::T
    function Traces(; kw...)
        for (k, v) in kw
            if !(v isa AbstractVector)
                throw(ArgumentError("the value of $k should be an AbstractVector"))
            end
        end

        data = map(x -> convert(LastDimSlices, x), values(kw))
        t = StructArray(data)
        new{typeof(t),keys(data),Tuple{typeof(data).types...}}(t)
    end
end

@forward Traces.traces Base.size, Base.parent, Base.getindex, Base.setindex!, Base.view, Base.push!, Base.pushfirst!, Base.pop!, Base.popfirst!, Base.empty!

Base.append!(t::Traces, x::NamedTuple) = append!(t.traces, StructArray(x))
Base.prepend!(t::Traces, x::NamedTuple) = prepend!(t.traces, StructArray(x))
Base.getindex(t::Traces, s::Symbol) = getproperty(t.traces, s)

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

function MultiplexTraces{names}(t) where {names}
    if length(names) != 2
        throw(ArgumentError("MultiplexTraces has exactly two sub traces, got $length(names) trace names"))
    end
    trace = convert(LastDimSlices, t)
    MultiplexTraces{names,typeof(trace),eltype(trace)}(trace)
end

function Base.getindex(t::MultiplexTraces{names}, k::Symbol) where {names}
    a, b = names
    if k == a
        @view t.trace[1:end-1]
    elseif k == b
        @view t.trace[2:end]
    else
        throw(ArgumentError("unknown trace name: $k"))
    end
end

Base.getindex(t::MultiplexTraces{names}, I::Int) where {names} = NamedTuple{names}(t[k][I] for k in names)
Base.getindex(t::MultiplexTraces{names}, I) where {names} = StructArray(NamedTuple{names}(t[k][I] for k in names))
Base.view(t::MultiplexTraces{names}, I) where {names} = StructArray(NamedTuple{names}(view(t[k], I) for k in names))
Base.size(t::MultiplexTraces) = (max(0, length(t.trace) - 1),)

function Base.setindex!(t::MultiplexTraces{names}, v::NamedTuple, i) where {names}
    a, b = names
    va, vb = getindex(v, a), getindex(v, b)
    t.trace[i] = va
    t.trace[i+1] = vb
end

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
    MergedTraces{ks,typeof(ts),length(k1) + length(k2),Tuple{T1.types...,T2.types...}}(ts, inds)
end

function Base.:(+)(t1::AbstractTraces{k1,T1}, t2::MergedTraces{k2,T,N,T2}) where {k1,T1,k2,T,N,T2}
    ks = (k1..., k2...)
    ts = (t1, t2.traces...)
    inds = merge(NamedTuple(k => 1 for k in k1), map(v => v + 1, t1.inds))
    MergedTraces{ks,typeof(ts),length(k1) + length(k2),Tuple{T1.types...,T2.types...}}(ts, inds)
end


function Base.:(+)(t1::MergedTraces{k1,T,N,T1}, t2::AbstractTraces{k2,T2}) where {k1,T,N,T1,k2,T2}
    ks = (k1..., k2...)
    ts = (t1.traces..., t2)
    inds = merge(t1.inds, (; (k => length(ts) for k in k2)...))
    MergedTraces{ks,typeof(ts),length(k1) + length(k2),Tuple{T1.types...,T2.types...}}(ts, inds)
end

function Base.:(+)(t1::MergedTraces{k1,T1,N1,E1}, t2::MergedTraces{k2,T2,N2,E2}) where {k1,T1,N1,E1,k2,T2,N2,E2}
    ks = (k1..., k2...)
    ts = (t1.traces..., t2.traces...)
    inds = merge(t1.inds, map(x -> x + length(t1.traces), t2.inds))
    MergedTraces{ks,typeof(ts),length(k1) + length(k2),Tuple{T1.types...,T2.types...}}(ts, inds)
end


Base.size(t::MergedTraces) = size(t.traces[1])
Base.getindex(t::MergedTraces, I::Int) = mapreduce(x -> getindex(x, I), merge, t.traces)
Base.getindex(t::MergedTraces, I) = StructArray(mapreduce(x -> getfield(getindex(x, I), :components), merge, t.traces))
Base.view(t::MergedTraces, I) = StructArray(mapreduce(x -> getfield(view(x, I), :components), merge, t.traces))

function Base.setindex!(t::MergedTraces, x::NamedTuple, I)
    for (k, v) in pairs(x)
        setindex!(t.traces[t.inds[k]], (; k => v), I)
    end
end


for f in (:push!, :pushfirst!, :append!, :prepend!)
    @eval function Base.$f(ts::MergedTraces, xs::NamedTuple)
        for (k, v) in pairs(xs)
            t = ts.traces[ts.inds[k]]
            $f(t, (; k => v))
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
