import MLUtils

MLUtils.batch(x::AbstractArray{<:Number}) = x

#####

import StackViews: StackView

lazy_stack(x) = StackView(x)
lazy_stack(x::AbstractVector{<:Number}) = x