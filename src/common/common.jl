export SS, LL, AA, RT, SSART, SSAART, SSLLAART

using CircularArrayBuffers

const SS = (:state, :next_state)
const LL = (:legal_actions_mask, :next_legal_actions_mask)
const AA = (:action, :next_action)
const RT = (:reward, :terminal)
const SSART = (SS..., :action, RT...)
const SSAART = (SS..., AA..., RT...)
const SSLART = (SS..., :legal_actions_mask, :action, RT...)
const SSLLAART = (SS..., LL..., AA..., RT...)

include("sum_tree.jl")
include("CircularArraySARTTraces.jl")
include("CircularArraySLARTTraces.jl")
