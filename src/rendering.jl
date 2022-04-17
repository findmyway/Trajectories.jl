using Term

TRACE_COLORS = ("bright_green", "hot_pink", "bright_blue", "light_coral", "bright_cyan", "sandy_brown", "violet")

# watch https://github.com/FedeClaudi/Term.jl/issues/78
Base.show(io::IO, t::Union{Trace,Traces}) = print(convert(Term.AbstractRenderable, t))

function inner_convert(::Type{Term.AbstractRenderable}, s::String; style="gray1", max_width=8)
    lines = split(s, "\n")
    max_len = mapreduce(length, max, lines)
    Panel(
        join((length(x) > max_width ? "$(x[1:max_width-3])..." : x for x in lines), "\n"),
        width=min(max_len,
            max_width) + 6,
        style=style,
        justify=:center
    )
end

inner_convert(t::Type{Term.AbstractRenderable}, x::Union{Symbol,Number}; kw...) = inner_convert(t, string(x); kw...)

function inner_convert(::Type{Term.AbstractRenderable}, x::AbstractArray; style="gray1")
    t = string(nameof(typeof(x)))
    s = replace(string(size(x)), " " => "")
    Panel(t * "\n" * s, width=max(length(s), length(t)) + 6, style=style, justify=:center)
end

function inner_convert(::Type{Term.AbstractRenderable}, x; style="gray1")
    s = string(nameof(typeof(x)))
    Panel(s, width=length(s) + 6, style=style, justify=:center)
end

Base.convert(T::Type{Term.AbstractRenderable}, t::Trace{<:AbstractArray}; kw...) = convert(T, Trace(collect(eachslice(t.x, dims=ndims(t.x)))); kw..., type=typeof(t), subtitle="size: $(size(t.x))")

function Base.convert(
    ::Type{Term.AbstractRenderable},
    t::Trace{<:AbstractVector};
    n_head=2,
    n_tail=1,
    name="Trace",
    style=TRACE_COLORS[mod1(hash(name), length(TRACE_COLORS))],
    type=typeof(t),
    subtitle="size: $(size(t.x))"
)
    "size: $(size(t.x))"
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
    Panel(content, title="$name: [italic $style]$type[/italic $style]", subtitle=subtitle, subtitle_justify=:right, style=style, subtitle_style="yellow")
end

function Base.convert(::Type{Term.AbstractRenderable}, t::Traces)
    max_len = mapreduce(length, max, t.traces)
    min_len = mapreduce(length, min, t.traces)
    if max_len - min_len == 1
        n_tails = [length(x) == max_len ? 2 : 1 for x in t.traces]
    else
        n_tails = [1 for x in t.traces]
    end
    Panel(
        "\n" / mapreduce(((i, x),) -> convert(Term.AbstractRenderable, t[x]; name=x, n_tail=n_tails[i], style=TRACE_COLORS[mod1(i, length(TRACE_COLORS))]) / "\n", /, enumerate(keys(t))),
        title="Traces",
        style="yellow"
    )
end
