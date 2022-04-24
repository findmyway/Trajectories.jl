export CircularArrayPISARTTraces

const CircularArrayPISARTTraces = Traces{
    PISART,
    <:Tuple{
        <:Trace{<:SumTree},
        <:Trace{<:CircularArrayBuffer},
        <:Trace{<:CircularArrayBuffer},
        <:Trace{<:CircularArrayBuffer},
        <:Trace{<:CircularArrayBuffer},
        <:Trace{<:CircularArrayBuffer},
    }
}


function CircularArrayPISARTTraces(;
    capacity::Int,
    priority=SumTree(capacity),
    state=Int => (),
    action=Int => (),
    reward=Float32 => (),
    terminal=Bool => ()
)
    state_eltype, state_size = state
    action_eltype, action_size = action
    reward_eltype, reward_size = reward
    terminal_eltype, terminal_size = terminal

    Traces(
        priority=priority,
        id=CircularArrayBuffer{Int}(capacity + 1),
        state=CircularArrayBuffer{state_eltype}(state_size..., capacity + 1),  # !!! state is one step longer
        action=CircularArrayBuffer{action_eltype}(action_size..., capacity + 1),  # !!! action is one step longer
        reward=CircularArrayBuffer{reward_eltype}(reward_size..., capacity),
        terminal=CircularArrayBuffer{terminal_eltype}(terminal_size..., capacity),
    )
end

function Random.rand(s::BatchSampler, t::CircularArrayPISARTTraces)
    inds, priority = rand(s.rng, t[:priority], s.batch_size)
    inds′ = inds .+ 1
    (
        priority=priority,
        state=t[:state][inds],
        action=t[:action][inds],
        reward=t[:reward][inds],
        terminal=t[:terminal][inds],
        next_state=t[:state][inds′],
        next_action=t[:state][inds′]
    )
end
