export AsyncTrajectory, UPDATE, SAMPLE, NO_OP

struct Update end
const UPDATE = Update()
struct SampleDecision end
const SAMPLE = SampleDecision()
struct NoOp end
const NO_OP = NoOp()


"""
    RateLimiter(n_samping_per_update=0.25; min_sampling_length=0, buffer_range=0:0)

- `n_samping_per_update`, a positive number, float number is supported.
- `min_sampling_length`, minimal number of elements to start sampling.
- `buffer_range`, if a single number is provided, it will be transformed into
  `-buffer_range:buffer_range`. Once the `trajectory` reaches
  `min_sampling_length`, if the length of `trajectory` is greater than the upper
  bound plus `min_sampling_length`, the `rate_limiter` always return `SAMPLE`.
  While if the length of `trajectory` is less than `min_sampling_length` minus
  the lower bound of `buffer_range`, then the `rate_limiter` always return
  `UPDATE`. In other cases, whether to `SAMPLE` or `UPDATE` depends on the
  availability of `in_channel` or `out_channel`.
"""
struct RateLimiter
    n_samping_per_update::Float64
    min_sampling_length::UInt
    buffer_range::UnitRange{Int}
    is_min_sampling_length_reached::Ref{Bool}
end

RateLimiter(n_samping_per_update=0.25; min_sampling_length=0, buffer_range=0) = RateLimiter(n_samping_per_update, min_sampling_length, buffer_range)
RateLimiter(n_samping_per_update, min_sampling_length, buffer_range::Number) = RateLimiter(n_samping_per_update, convert(UInt, min_sampling_length), range(-buffer_range, buffer_range))
RateLimiter(n_samping_per_update, min_sampling_length, buffer_range) = RateLimiter(n_samping_per_update, min_sampling_length, buffer_range, Ref(false))

"""
    (r::RateLimiter)(n_update, n_sample, is_in_ready, is_out_ready)

- `n_update`, number of elements inserted into `trajectory`. 
- `n_sample`, number of batches sampled from `trajectory`.
- `is_in_ready`, the `in_channel` has elements to put into `trajectory` or not. 
- `is_out_ready`, the `out_channel` is ready to consume new samplings or not.
"""
function (r::RateLimiter)(n_update, n_sample, is_in_ready, is_out_ready)
    if n_update >= r.min_sampling_length
        r.is_min_sampling_length_reached[] = true
    end

    if r.is_min_sampling_length_reached[]
        n_estimated_updates = n_sample / r.n_samping_per_update + r.min_sampling_length
        if n_estimated_updates < n_update + r.buffer_range[begin]
            SAMPLE
        elseif n_estimated_updates > n_update + r.buffer_range[end]
            UPDATE
        else
            if is_out_ready
                SAMPLE
            elseif is_in_ready
                UPDATE
            else
                NO_OP
            end
        end
    else
        UPDATE
    end
end

#####

struct CallMsg
    f::Any
    args::Tuple
    kw::Any
end

struct AsyncTrajectory
    trajectory
    sampler
    rate_limiter
    channel_in
    channel_out
    task
    n_update_ref
    n_sample_ref

    function AsyncTrajectory(trajectory, sampler, rate_limiter; channel_in=Channel(1), channel_out=Channel(1))
        n_update_ref = Ref(0)
        n_sample_ref = Ref(0)
        task = @async while true
            decision = rate_limiter(n_update, n_sample, isready(channel_in), Base.n_avail(channel_out) < length(channel_out.data))
            if decision === UPDATE
                msg = take!(channel_in)
                if msg.f === Base.push!
                    n_pre = length(trajectory)
                    push!(trajectory, msg.args...; msg.kw...)
                    n_post = length(trajectory)
                    n_update_ref[] += n_post - n_pre
                elseif msg.f === Base.append!
                    n_pre = length(trajectory)
                    append!(trajectory, msg.args...; msg.kw...)
                    n_post = length(trajectory)
                    n_update_ref[] += n_post - n_pre
                else
                    msg.f(trajectory, msg.args...; msg.kw...)
                end
            elseif decision === SAMPLE
                put!(channel_out, rand(sampler, trajectory))
                n_sample_ref[] += 1
            end
        end
        new(
            trajectory,
            sampler,
            channel_in,
            channel_out,
            task,
            n_update_ref,
            n_sample_ref
        )
    end
end

Base.push!(t::AsyncTrajectory, args...; kw...) = put!(t.in, CallMsg(Base.push!, args, kw))
Base.append!(t::AsyncTrajectory, args...; kw...) = put!(t.in, CallMsg(Base.append!, args, kw))
Base.take!(t::AsyncTrajectory) = take!(t.out)
