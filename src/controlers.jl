export InsertSampleRatioControler, InsertSampleControler, AsyncInsertSampleRatioControler

mutable struct InsertSampleRatioControler
    ratio::Float64
    threshold::Int
    n_inserted::Int
    n_sampled::Int
end

"""
    InsertSampleRatioControler(ratio, threshold)

Used in [`Trajectory`](@ref). The `threshold` means the minimal number of
insertings before sampling. The `ratio` balances the number of insertings and
the number of samplings.
"""
InsertSampleRatioControler(ratio, threshold) = InsertSampleRatioControler(ratio, threshold, 0, 0)

function on_insert!(c::InsertSampleRatioControler, n::Int)
    if n > 0
        c.n_inserted += n
    end
end

function on_sample!(c::InsertSampleRatioControler)
    if c.n_inserted >= c.threshold
        if c.n_sampled <= (c.n_inserted - c.threshold) * c.ratio
            c.n_sampled += 1
            true
        end
    end
end

"""
    InsertSampleControler(n, threshold)

Used in [`Trajectory`](@ref). The `threshold` means the minimal number of
insertings before sampling. The `n` is the number of samples until stopping.
"""
mutable struct InsertSampleControler
    n::Int
    threshold::Int
    n_inserted::Int
    n_sampled::Int
end

InsertSampleControler(n, threshold) = InsertSampleControler(n, threshold, 0, 0)

function on_insert!(c::InsertSampleControler, n::Int)
    if n > 0
        c.n_inserted += n
    end
end

function on_sample!(c::InsertSampleControler)
    if c.n_inserted >= c.threshold
        if c.n_sampled < c.n
            c.n_sampled += 1
            true
        end
    end
end

#####

mutable struct AsyncInsertSampleRatioControler
    ratio::Float64
    threshold::Int
    n_inserted::Int
    n_sampled::Int
    ch_in::Channel
    ch_out::Channel
end

function AsyncInsertSampleRatioControler(
    ratio,
    threshold,
    ; ch_in_sz=1,
    ch_out_sz=1,
    n_inserted=0,
    n_sampled=0
)
    AsyncInsertSampleRatioControler(
        ratio,
        threshold,
        n_inserted,
        n_sampled,
        Channel(ch_in_sz),
        Channel(ch_out_sz)
    )
end
