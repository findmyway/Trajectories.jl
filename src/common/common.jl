using CircularArrayBuffers

const SART = (:state, :action, :reward, :terminal)
const SARTSA = (:state, :action, :reward, :terminal, :next_state, :next_action)
const SLART = (:state, :legal_actions_mask, :action, :reward, :terminal)
const SLARTSLA = (:state, :legal_actions_mask, :action, :reward, :terminal, :next_state, :next_legal_actions_mask, :next_action)

include("sum_tree.jl")
include("CircularArraySARTTraces.jl")
include("CircularArraySLARTTraces.jl")
