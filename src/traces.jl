export Trace, Traces

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
Base.length(t::Trace{<:AbstractArray}) = size(t.x, ndims(t.x))

Base.lastindex(t::Trace) = length(t)
Base.firstindex(t::Trace) = 1

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

##

function Random.rand(s::BatchSampler, t::Trace)
    inds = rand(s.rng, 1:length(t), s.batch_size)
    t[inds]
end

#####

"""
    Traces(;kw...)

A container of several named-[`Trace`](@ref)s. Each element in the `kw` will be converted into a `Trace`.
"""
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
Base.length(t::Traces) = mapreduce(length, min, t.traces)

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

##
function Random.rand(s::BatchSampler, t::Traces)
    inds = rand(s.rng, 1:length(t), s.batch_size)
    map(t.traces) do x
        x[inds]
    end
end