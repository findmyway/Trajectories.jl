export CircularArrayPSLARTTraces

const CircularArrayPSLARTTraces = Traces{
    PSLART,
    <:Tuple{
        <:Trace{<:SumTree},
        <:Trace{<:CircularArrayBuffer},
        <:Trace{<:CircularArrayBuffer},
        <:Trace{<:CircularArrayBuffer},
        <:Trace{<:CircularArrayBuffer},
        <:Trace{<:CircularArrayBuffer}
    }
}


function CircularArrayPSLARTTraces(;
    capacity::Int,
    priority=SumTree(capacity),
    state=Int => (),
    legal_actions_mask=Bool => (),
    action=Int => (),
    reward=Float32 => (),
    terminal=Bool => ()
)
    state_eltype, state_size = state
    action_eltype, action_size = action
    legal_actions_mask_eltype, legal_actions_mask_size = legal_actions_mask
    reward_eltype, reward_size = reward
    terminal_eltype, terminal_size = terminal

    Traces(
        priority=priority,
        state=CircularArrayBuffer{state_eltype}(state_size..., capacity + 1),  # !!! state is one step longer
        legal_actions_mask=CircularArrayBuffer{legal_actions_mask_eltype}(legal_actions_mask_size..., capacity + 1),  # !!! legal_actions_mask is one step longer
        action=CircularArrayBuffer{action_eltype}(action_size..., capacity + 1),  # !!! action is one step longer
        reward=CircularArrayBuffer{reward_eltype}(reward_size..., capacity),
        terminal=CircularArrayBuffer{terminal_eltype}(terminal_size..., capacity),
    )
end