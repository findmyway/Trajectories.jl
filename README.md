# Trajectories

[![Build Status](https://github.com/JuliaReinforcementLearning/Trajectories.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaReinforcementLearning/Trajectories.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaReinforcementLearning/Trajectories.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaReinforcementLearning/Trajectories.jl)
[![PkgEval](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/T/Trajectories.svg)](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/report.html)

## Core Concepts

### Install & Usage

```julia
julia> ] add Trajectories

julia> using Trajectories
```

### Trace

```julia
Trace(data)
```

A wrapper of arbitrary container. Generally we assume the `data` is an
`AbstractVector` like object. When an `AbstractArray` is given, we view it as a
vector of sub-arrays along the last dimension.

```julia
t = Trace([1,2,3])
```


## Design

```
      ┌────────────────────────────┐
      │(state=..., action=..., ...)│
      └──────────────┬─────────────┘
              push!  │  append!
 ┌───────────────────▼───────────────────┐
 │ Trajectory                            │
 │  ┌─────────────────────────────────┐  │
 │  │ Traces                          │  │
 │  │          ┌───────────────────┐  │  │
 │  │   state: │CircularArrayBuffer│  │  │
 │  │          └───────────────────┘  │  │
 │  │          ┌───────────────────┐  │  │
 │  │   action:│CircularArrayBuffer│  │  │
 │  │          └───────────────────┘  │  │
 │  │   ......                        │  │
 │  └─────────────────────────────────┘  │
 |    Sampler                            |
 └───────────────────┬───────────────────┘
                     │ batch sampling
      ┌──────────────▼─────────────┐
      │(state=..., action=..., ...)│
      └────────────────────────────┘
```

```
 ┌──────────────┐    ┌──────────────┐
 │Single Element│    │Batch Elements│
 └──────┬───────┘    └──────┬───────┘
        │                   │
  push! └──────┐    ┌───────┘ append!
               │    │
 ┌─────────────┼────┼─────────────────────────────┐
 │          ┌──▼────▼──┐     AsyncTrajectory      │
 │          │Channel In│                          │
 │          └─────┬────┘                          │
 │          take! │                               │
 │          ┌─────▼─────┐   push!  ┌────────────┐ │
 │          │RateLimiter├──────────► Trajectory │ │
 │          └─────┬─────┘  append! └────*───────┘ │
 │                │                     *         │
 │           put! │**********************         │
 │                │     batch sampling            │
 │          ┌─────▼─────┐                         │
 │          │Channel Out│                         │
 │          └───────────┘                         │
 └────────────────────────────────────────────────┘
```

## Acknowledgement

This async version is mainly inspired by [deepmind/reverb](https://github.com/deepmind/reverb). 
