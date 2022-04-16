export Trace, Traces

using Term
using CircularArrayBuffers

TRACE_COLORS = ("bright_green", "hot_pink", "bright_blue", "light_coral", "bright_cyan", "sandy_brown", "violet")

struct Trace{T}
    x::T
end

Base.convert(::Type{Trace}, x) = Trace(x)

# watch https://github.com/FedeClaudi/Term.jl/issues/78
Base.show(io::IO, t::Trace) = redirect_stdout(() -> print(convert(Term.AbstractRenderable, t)), io)

function inner_convert(::Type{Term.AbstractRenderable}, x::Number; style="gray1")
    s = string(x)
    if length(s) >= 7
        s = "$(s[1:7])..."
    end
    Panel(s, width=length(s) + 6, style=style)
end

function inner_convert(::Type{Term.AbstractRenderable}, x::AbstractArray; style="gray1")
    t = string(nameof(typeof(x)))
    s = string(size(x))
    Panel(t * "\n" * s, width=max(length(s), length(t)) + 6, style=style)
end

function inner_convert(::Type{Term.AbstractRenderable}, x; style="gray1")
    s = string(nameof(typeof(x)))
    Panel(s, width=length(s) + 6, style=style)
end

function Base.convert(::Type{Term.AbstractRenderable}, t::Trace{<:CircularArrayBuffer}; n_head=2, n_tail=1, title="Trace", style=TRACE_COLORS[mod1(hash(title), length(TRACE_COLORS))])
    n = size(t.x)[end]
    describe = "$(size(t.x)[1:end-1])"
    if n == 0
        content = ""
    elseif 1 <= n <= n_head + n_tail
        content = mapreduce(x -> Panel("SubArray\n$describe", style=style, width=length(describe) + 6), *, 1:n)
    else
        content = mapreduce(x -> Panel("SubArray\n$describe", style=style, width=length(describe) + 6), *, 1:n_head) *
                  TextBox("...", width=9, justify=:center) *
                  mapreduce(x -> Panel("SubArray\n$describe", style=style, width=length(describe) + 6), *, 1:n_tail)
    end
    Panel(content, title="$title: [italic $style]$(typeof(t))[/italic $style]", subtitle="size: $(size(t.x))", subtitle_justify=:right, style=style, subtitle_style="yellow")
end

function Base.convert(::Type{Term.AbstractRenderable}, t::Trace{<:AbstractVector}; n_head=2, n_tail=1, title="Trace", style=TRACE_COLORS[mod1(hash(title), length(TRACE_COLORS))])
    n = length(t.x)
    if n == 0
        content = ""
    elseif 1 <= n <= n_head + n_tail
        content = mapreduce(x -> inner_convert(Term.AbstractRenderable, x, style=style), *, t.x)
    else
        content = mapreduce(x -> inner_convert(Term.AbstractRenderable, x, style=style), *, t.x[1:n_head]) *
                  TextBox("...", width=9, justify=:center) *
                  mapreduce(x -> inner_convert(Term.AbstractRenderable, x, style=style), *, t.x[end-n_tail+1:end])
    end
    Panel(content, title="$title: [italic $style]$(typeof(t))[/italic $style]", subtitle="size: $(size(t.x))", subtitle_justify=:right, style=style, subtitle_style="yellow")
end

struct Traces{names,T}
    traces::NamedTuple{names,T}
    function Traces(; kw...)
        traces = map(x -> convert(Trace, x), values(kw))
        new{keys(traces),typeof(values(traces))}(traces)
    end
end

# watch https://github.com/FedeClaudi/Term.jl/issues/78
Base.show(io::IO, t::Traces) = redirect_stdout(() -> print(convert(Term.AbstractRenderable, t)), io)

function Base.convert(::Type{Term.AbstractRenderable}, t::Traces)
    Panel(
        "\n" / mapreduce(((i, x),) -> convert(Term.AbstractRenderable, t[x]; title=x, style=TRACE_COLORS[mod1(i, length(TRACE_COLORS))]) / "\n", /, enumerate(keys(t))),
        title="Traces",
        style="yellow"
    )
end

Base.keys(t::Traces) = keys(t.traces)
Base.haskey(t::Traces, s::Symbol) = haskey(t.traces, s)
Base.getindex(t::Traces, x) = getindex(t.traces, x)

function Base.push!(t::Traces, x::NamedTuple)
    for k in keys(x)
        push!(t[k], x[k])
    end
end

function Base.append!(t::Traces, x::NamedTuple)
    for k in keys(x)
        append!(t[k], x[k])
    end
end

Base.pop!(t::Traces) = map(pop!, t.traces)
Base.empty!(t::Traces) = map(empty!, t.traces)
