using CircularArrayBuffers

const SART = (:state, :action, :reward, :terminal)
const SLART = (:state, :legal_actions_mask, :action, :reward, :terminal)
const PSART = (:priority, :state, :action, :reward, :terminal)
const PSLART = (:priority, :state, :legal_actions_mask, :action, :reward, :terminal)
const PISART = (:priority, :id, :state, :action, :reward, :terminal)
const PISLART = (:priority, :id, :state, :legal_actions_mask, :action, :reward, :terminal)