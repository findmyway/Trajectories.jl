export LastDimSlices

using MacroTools: @forward

# See also https://github.com/JuliaLang/julia/pull/32310

struct LastDimSlices{T,E} <: AbstractVector{E}
    parent::T
end

function LastDimSlices(x::T) where {T<:AbstractArray}
    E = eltype(x)
    N = ndims(x) - 1
    P = typeof(x)
    I = Tuple{ntuple(_ -> Base.Slice{Base.OneTo{Int}}, Val(ndims(x) - 1))...,Int}
    LastDimSlices{T,SubArray{E,N,P,I,true}}(x)
end

Base.convert(::Type{LastDimSlices}, x::AbstractVector) = x
Base.convert(::Type{LastDimSlices}, x::AbstractArray) = LastDimSlices(x)

Base.size(x::LastDimSlices) = (size(x.parent, ndims(x.parent)),)
Base.getindex(s::LastDimSlices, I) = getindex(s.parent, ntuple(i -> i == ndims(s.parent) ? I : (:), Val(ndims(s.parent)))...)
Base.view(s::LastDimSlices, I) = view(s.parent, ntuple(i -> i == ndims(s.parent) ? I : (:), Val(ndims(s.parent)))...)
Base.setindex!(s::LastDimSlices, v, I) = setindex!(s.parent, v, ntuple(i -> i == ndims(s.parent) ? I : (:), Val(ndims(s.parent)))...)

@forward LastDimSlices.parent Base.parent, Base.pushfirst!, Base.push!, Base.pop!, Base.append!, Base.prepend!, Base.empty!