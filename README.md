# Trajectories

[![Build Status](https://github.com/JuliaReinforcementLearning/Trajectories.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaReinforcementLearning/Trajectories.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaReinforcementLearning/Trajectories.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaReinforcementLearning/Trajectories.jl)
[![PkgEval](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/T/Trajectories.svg)](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/report.html)

## Design

The relationship of several concepts provided in this package:

```
┌───────────────────────────────────┐
│ Trajectory                        │
│ ┌───────────────────────────────┐ │
│ │ AbstractTraces                │ │
│ │             ┌───────────────┐ │ │
│ │ :trace_A => │ AbstractTrace │ │ │
│ │             └───────────────┘ │ │
│ │                               │ │
│ │             ┌───────────────┐ │ │
│ │ :trace_B => │ AbstractTrace │ │ │
│ │             └───────────────┘ │ │
│ │  ...             ...          │ │
│ └───────────────────────────────┘ │
│          ┌───────────┐            │
│          │  Sampler  │            │
│          └───────────┘            │
│         ┌────────────┐            │
│         │ Controller │            │
│         └────────────┘            │
└───────────────────────────────────┘
```

## `Trajectory`

A `Trajectory` contains 3 parts:

- A `container` to store data. (Usually an `AbstractTraces`)
- A `sampler` to determine how to sample a batch from `container`
- A `controller` to decide when to sample a new batch from the `container`

Typical usage:

```julia
julia> t = Trajectory(Traces(a=Int[], b=Bool[]), BatchSampler(3), InsertSampleRatioControler(1.0, 3));

julia> for i in 1:5
           push!(t, (a=i, b=iseven(i)))
       end

julia> for batch in t
           println(batch)
       end
(a = [4, 5, 1], b = Bool[1, 0, 0])
(a = [3, 2, 4], b = Bool[0, 1, 1])
(a = [4, 1, 2], b = Bool[1, 0, 1])
```

### `AbstractTrace`

`Trace` is the most commonly used `AbstractTrace`. It provides a sequential view on other containers.

```julia
julia> t = Trace([1,2,3])
3-element Trace{Vector{Int64}, SubArray{Int64, 0, Vector{Int64}, Tuple{Int64}, true}}:
 1
 2
 3
julia> push!(t, 4)
4-element Vector{Int64}:
 1
 2
 3
 4

julia> append!(t, 5:6)
6-element Vector{Int64}:
 1
 2
 3
 4
 5
 6

julia> pop!(t)
6

julia> popfirst!(t)
1

julia> t
4-element Trace{Vector{Int64}, SubArray{Int64, 0, Vector{Int64}, Tuple{Int64}, true}}:
 2
 3
 4
 5

julia> empty!(t)
Int64[]

julia> t
0-element Trace{Vector{Int64}, SubArray{Int64, 0, Vector{Int64}, Tuple{Int64}, true}}
```

In most cases, it's just the same with a `Vector`.

When an `AbstractArray` with higher dimension provided, it is **slice**d along the last dimension to provide a sequential view.

```julia
julia> t = Trace(rand(2,3))
3-element Trace{Matrix{Float64}, SubArray{Float64, 1, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true}}:
 [0.276012181224494, 0.6621365818458671]
 [0.9937726056924112, 0.3308302850028162]
 [0.9856543000075456, 0.6123660950650406]

julia> t[1]
2-element view(::Matrix{Float64}, :, 1) with eltype Float64:
 0.276012181224494
 0.6621365818458671

julia> t[1] = [0., 1.]
2-element Vector{Float64}:
 0.0
 1.0

julia> t
3-element Trace{Matrix{Float64}, SubArray{Float64, 1, Matrix{Float64}, Tuple{Base.Slice{Base.OneTo{Int64}}, Int64}, true}}:
 [0.0, 1.0]
 [0.9937726056924112, 0.3308302850028162]
 [0.9856543000075456, 0.6123660950650406]

julia> t[[2,3,1]]
2×3 view(::Matrix{Float64}, :, [2, 3, 1]) with eltype Float64:
 0.993773  0.985654  0.0
 0.33083   0.612366  1.0
```

**Note** that when indexing a `Trace`, a **view** is returned. As you can see above, the data is modified in-place.

### `AbstractTraces`

`Traces` is one of the common `AbstractTraces`. It is similar to a `NamedTuple` of several traces.

```julia
julia> t = Traces(;
           a=[1, 2],
           b=Bool[0, 1]
       )  # note that `a` and `b` are converted into `Trace` implicitly
Traces with 2 traces:
  :a => 2-element Trace{Vector{Int64}, SubArray{Int64, 0, Vector{Int64}, Tuple{Int64}, true}}
  :b => 2-element Trace{Vector{Bool}, SubArray{Bool, 0, Vector{Bool}, Tuple{Int64}, true}}


julia> push!(t, (a=3, b=false))

julia> t
Traces with 2 traces:
  :a => 3-element Trace{Vector{Int64}, SubArray{Int64, 0, Vector{Int64}, Tuple{Int64}, true}}
  :b => 3-element Trace{Vector{Bool}, SubArray{Bool, 0, Vector{Bool}, Tuple{Int64}, true}}


julia> t[:a]
3-element Trace{Vector{Int64}, SubArray{Int64, 0, Vector{Int64}, Tuple{Int64}, true}}:
 1
 2
 3

julia> t[:b]
3-element Trace{Vector{Bool}, SubArray{Bool, 0, Vector{Bool}, Tuple{Int64}, true}}:
 false
  true
 false

julia> t[1]
(a = 1, b = false)

julia> t[1:3]
(a = [1, 2, 3], b = Bool[0, 1, 0])
```

Another commonly used traces is `MultiplexTraces`. In reinforcement learning, *states* and *next-states* share most data except for the first and last element.

```julia
julia> t = MultiplexTraces{(:state, :next_state)}([1,2,3]);

julia> t[:state]
2-element Trace{SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}, SubArray{Int64, 0, SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}, Tuple{Int64}, true}}:
 1
 2

julia> t[:next_state]
2-element Trace{SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}, SubArray{Int64, 0, SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}, Tuple{Int64}, true}}:
 2
 3

julia> push!(t, (;state=4))
4-element Vector{Int64}:
 1
 2
 3
 4

julia> t[:state]
3-element Trace{SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}, SubArray{Int64, 0, SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}, Tuple{Int64}, true}}:
 1
 2
 3

julia> t[:next_state]
3-element Trace{SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}, SubArray{Int64, 0, SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}, Tuple{Int64}, true}}:
 2
 3
 4

julia> length(t)
3
```

Note that different kinds of `AbstractTraces` can be combined to form a `MergedTraces`.

```
ulia> t1 = Traces(a=Int[])
       t2 = MultiplexTraces{(:b, :c)}(Int[])
       t3 = t1 + t2
MergedTraces with 3 traces:
  :a => 0-element Trace{Vector{Int64}, SubArray{Int64, 0, Vector{Int64}, Tuple{Int64}, true}}
  :b => 0-element Trace{SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}, SubArray{Int64, 0, SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}, Tuple{Int64}, true}}
  :c => 0-element Trace{SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}, SubArray{Int64, 0, SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}, Tuple{Int64}, true}}


julia> push!(t3, (a=1,b=2,c=3))

julia> t3[:a]
1-element Trace{Vector{Int64}, SubArray{Int64, 0, Vector{Int64}, Tuple{Int64}, true}}:
 1

julia> t3[:b]
1-element Trace{SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}, SubArray{Int64, 0, SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}, Tuple{Int64}, true}}:
 2

julia> t3[:c]
1-element Trace{SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}, SubArray{Int64, 0, SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}, Tuple{Int64}, true}}:
 3

julia> push!(t3, (a=-1, b=-2))

julia> t3[:a]
2-element Trace{Vector{Int64}, SubArray{Int64, 0, Vector{Int64}, Tuple{Int64}, true}}:
  1
 -1

julia> t3[:b]
2-element Trace{SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}, SubArray{Int64, 0, SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}, Tuple{Int64}, true}}:
 2
 3

julia> t3[:c]
2-element Trace{SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}, SubArray{Int64, 0, SubArray{Int64, 1, Vector{Int64}, Tuple{UnitRange{Int64}}, true}, Tuple{Int64}, true}}:
  3
 -2
```
## Acknowledgement

This async version is mainly inspired by [deepmind/reverb](https://github.com/deepmind/reverb). 
