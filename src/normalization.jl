import OnlineStats: OnlineStats, Group, Moments, fit!, OnlineStat, Weight, EqualWeight, mean, std
export scalar_normalizer, array_normalizer, NormalizedTraces, Normalizer
import MacroTools.@forward

"""
    Normalizer(::OnlineStat)

Wraps an OnlineStat to be used by a [`NormalizedTraces`](@ref).
"""
struct Normalizer{OS<:OnlineStat}
    os::OS
end

@forward Normalizer.os OnlineStats.mean, OnlineStats.std, Base.iterate, normalize, Base.length

#Treats last dim as batch dim
function OnlineStats.fit!(n::Normalizer, data::AbstractArray)
    for d in eachslice(data, dims = ndims(data))
        fit!(n.os, vec(d))
    end
    n
end

function OnlineStats.fit!(n::Normalizer{<:Group}, y::AbstractVector)
    fit!(n.os, y)
    n
end

function OnlineStats.fit!(n::Normalizer, y)
    for yi in y
        fit!(n.os, vec(yi))
    end
    n
end

function OnlineStats.fit!(n::Normalizer{<:Moments}, y::AbstractVector{<:Number})
    for yi in y
        fit!(n.os, yi)
    end
    n
end

function OnlineStats.fit!(n::Normalizer, data::Number)
    fit!(n.os, data)
    n
end

"""
    normalize(os::Moments, x)

Given an Moments estimate of the elements of x, a vector of scalar traces,
normalizes x elementwise to zero mean, and unit variance. 
"""
function normalize(os::Moments, x)
    T = eltype(x)
    m, s = T(mean(os)), T(std(os))
    return (x .- m) ./ s
end

"""
    normalize(os::Group{<:AbstractVector{<:Moments}}, x)

Given an os::Group{<:Tuple{Moments}}, that is, a multivariate estimator of the moments of each element of x,
normalizes each element of x to zero mean, and unit variance. Treats the last dimension as a batch dimension if `ndims(x) >= 2`.
"""
function normalize(os::Group{<:AbstractVector{<:Moments}}, x::AbstractVector)
    T = eltype(x)
    m = [T(mean(stat)) for stat in os]
    s = [T(std(stat)) for stat in os]
    return (x .- m) ./ s
end

function normalize(os::Group{<:AbstractVector{<:Moments}}, x::AbstractArray) 
    xn = similar(x)
    for (i, slice) in enumerate(eachslice(x, dims = ndims(x)))
        xn[repeat([:], ndims(x)-1)..., i] .= reshape(normalize(os, vec(slice)), size(x)[1:end-1]...) 
    end
    return xn
end

function normalize(os::Group{<:AbstractVector{<:Moments}}, x::AbstractVector{<:AbstractArray})
    xn = similar(x)
    for (i,el) in enumerate(x)
        xn[i] = normalize(os, vec(el))
    end
    return xn
end

"""
    scalar_normalizer(;weights = OnlineStats.EqualWeight())

Returns preconfigured normalizer for scalar traces such as rewards. By default, all samples have equal weights in the computation of the moments.
See the [OnlineStats documentation](https://joshday.github.io/OnlineStats.jl/stable/weights/) to use variants such as exponential weights to favor the most recent observations.
"""
scalar_normalizer(; weight::Weight = EqualWeight()) = Normalizer(Moments(weight = weight))

"""
    array_normalizer(size::Tuple{Int}; weights = OnlineStats.EqualWeight())

Returns preconfigured normalizer for array traces such as vector or matrix states. 
`size` is a tuple containing the dimension sizes of a state. E.g. `(10,)` for a 10-elements vector, or `(252,252)` for a square image.
By default, all samples have equal weights in the computation of the moments.
See the [OnlineStats documentation](https://joshday.github.io/OnlineStats.jl/stable/weights/) to use variants such as exponential weights to favor the most recent observations.
"""
array_normalizer(size::NTuple{N,Int}; weight::Weight = EqualWeight()) where N = Normalizer(Group([Moments(weight = weight) for _ in 1:prod(size)]))

"""
    NormalizedTrace(trace::Trace, normalizer::Normalizer)

Wraps a [`Trace`](@ref) and a [`Normalizer`](@ref). When pushing new elements to the trace, a `NormalizedTrace` will first update a running estimate of the moments of that trace.
When sampling a normalized trace, it will first normalize the samples using to zero mean and unit variance.

preconfigured normalizers are provided for scalar (see [`scalar_normalizer`](@ref)) and arrays (see [`array_normalizer`](@ref))

#Example
t = Trajectory(
    container=Traces(
        a_scalar_trace = NormalizedTrace(Float32[], scalar_normalizer()),
        a_non_normalized_trace=Bool[],
        a_vector_trace = NormalizedTrace(Vector{Float32}[], array_normalizer((10,))),
        a_matrix_trace = NormalizedTrace(Matrix{Float32}[], array_normalizer((252,252), weight = OnlineStats.ExponientialWeight(0.9f0)))
    ),
    sampler=BatchSampler(3),
    controler=InsertSampleRatioControler(0.25, 4)
)

"""
struct NormalizedTraces{names, TT, T <: AbstractTraces{names, TT}, normnames, N} <: AbstractTraces{names, TT}
    traces::T
    normalizers::NamedTuple{normnames, N}
end 

function NormalizedTraces(traces::AbstractTraces{names, TT}; trace_normalizer_pairs...) where names where TT
    for key in keys(trace_normalizer_pairs)
        @assert key in keys(traces) "Traces do not have key $key, valid keys are $(keys(traces))."
    end
    nt = (; trace_normalizer_pairs...)
    NormalizedTraces{names, TT, typeof(traces), keys(nt), typeof(values(nt))}(traces, nt)
end

function Base.show(io::IO, ::MIME"text/plain", t::NormalizedTraces{names,T}) where {names,T}
    s = nameof(typeof(t))
    println(io, "$s with $(length(names)) entries:")
    for n in names
        print(io, "  :$n => $(summary(t[n]))")
        if n in keys(t.normalizers)
            println(io, " => Normalized")
        else
            println(io, "")
        end
    end
end

@forward NormalizedTraces.traces Base.length, Base.size, Base.lastindex, Base.firstindex, Base.getindex, Base.view, Base.pop!, Base.popfirst!, Base.empty!, Base.parent

for f in (:push!, :pushfirst!, :append!, :prepend!)
    @eval function Base.$f(nt::NormalizedTraces, x::NamedTuple)
        for key in intersect(keys(nt.normalizers), keys(x))
            fit!(nt.normalizers[key], x[key])
        end
        $f(nt.traces, x)
    end
end

function sample(s::BatchSampler, nt::NormalizedTraces, names)
    inds = rand(s.rng, 1:length(nt), s.batch_size)
    maybe_normalize(data, key) = key in keys(nt.normalizers) ? normalize(nt.normalizers[key], data) : data 
    NamedTuple{names}(s.transformer(maybe_normalize(nt[x][inds], x) for x in names))
end
