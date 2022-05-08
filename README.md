# Trajectories

[![Build Status](https://github.com/JuliaReinforcementLearning/Trajectories.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaReinforcementLearning/Trajectories.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaReinforcementLearning/Trajectories.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaReinforcementLearning/Trajectories.jl)
[![PkgEval](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/T/Trajectories.svg)](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/report.html)

## Design

A typical example of `Trajectory`:

```
╭──── Trajectory ──────────────────────────────────────────────────────────────────────────────────────────────────╮
│  ╭──── Traces ────────────────────────────────────────────────────────────────────────────────────────────────╮  │
│  │  ╭──── state: Trac... ────╮╭──── action: Tra... ────╮╭──── reward: Tra... ────╮╭──── terminal: T... ────╮  │  │
│  │  │  ╭──────────────────╮  ││  ╭──────────────────╮  ││  ╭──────────────────╮  ││  ╭──────────────────╮  │  │  │
│  │  │  │     SubArray     │  ││  │        1         │  ││  │       1.0        │  ││  │      false       │  │  │  │
│  │  │  │     (2,3)        │  ││  ╰──────────────────╯  ││  ╰──────────────────╯  ││  ╰──────────────────╯  │  │  │
│  │  │  ╰──────────────────╯  ││  ╭──────────────────╮  ││  ╭──────────────────╮  ││  ╭──────────────────╮  │  │  │
│  │  │  ╭──────────────────╮  ││  │        2         │  ││  │       2.0        │  ││  │      false       │  │  │  │
│  │  │  │     SubArray     │  ││  ╰──────────────────╯  ││  ╰──────────────────╯  ││  ╰──────────────────╯  │  │  │
│  │  │  │     (2,3)        │  ││  ╭──────────────────╮  ││  ╭──────────────────╮  ││  ╭──────────────────╮  │  │  │
│  │  │  ╰──────────────────╯  ││  │       ...        │  ││  │       ...        │  ││  │       ...        │  │  │  │
│  │  │  ╭──────────────────╮  ││  ╰──────────────────╯  ││  ╰──────────────────╯  ││  ╰──────────────────╯  │  │  │
│  │  │  │       ...        │  ││  ╭──────────────────╮  ││  ╭──────────────────╮  ││  ╭──────────────────╮  │  │  │
│  │  │  ╰──────────────────╯  ││  │        3         │  ││  │       3.0        │  ││  │       true       │  │  │  │
│  │  │  ╭──────────────────╮  ││  ╰──────────────────╯  ││  ╰──────────────────╯  ││  ╰──────────────────╯  │  │  │
│  │  │  │     SubArray     │  ││  ╭──────────────────╮  │╰───────── size: (4,) ───╯╰───────── size: (4,) ───╯  │  │
│  │  │  │     (2,3)        │  ││  │        3         │  │                                                      │  │
│  │  │  ╰──────────────────╯  ││  ╰──────────────────╯  │                                                      │  │
│  │  │  ╭──────────────────╮  │╰───────── size: (5,) ───╯                                                      │  │
│  │  │  │     SubArray     │  │                                                                                │  │
│  │  │  │     (2,3)        │  │                                                                                │  │
│  │  │  ╰──────────────────╯  │                                                                                │  │
│  │  ╰──── size: (2, 3, 5) ───╯                                                                                │  │
│  ╰────────────────────────────────────────────────────────────────────────────────────── 4 traces in total ───╯  │
│  ╭──── sampler ───────────────────────────────────────────────────────╮                                          │
│  │   BatchSampler                                                     │                                          │
│  │  ━━━━━━━━━━━━━━                                                    │                                          │
│  │        │                                                           │                                          │
│  │        ├── transformer => identity (generic function...: identity  │                                          │
│  │        ├── rng => Random._GLOBAL_RNG: Random._GLOBAL_RNG()         │                                          │
│  │        └── batch_size => Int64: 5                                  │                                          │
│  ╰────────────────────────────────────────────────────────────────────╯                                          │
│  ╭──── controler ────────────────────────────╮                                                                   │
│  │   InsertSampleRatioControler              │                                                                   │
│  │  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━             │                                                                   │
│  │               │                           │                                                                   │
│  │               ├── threshold => Int64: 4   │                                                                   │
│  │               ├── n_sampled => Int64: 0   │                                                                   │
│  │               ├── ratio => Float64: 0.25  │                                                                   │
│  │               └── n_inserted => Int64: 4  │                                                                   │
│  ╰───────────────────────────────────────────╯                                                                   │
╰──────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

Exported APIs are:

```julia
push!(trajectory; [trace_name=value]...)
append!(trajectory; [trace_name=value]...)

for sample in trajectory
    # consume samples from the trajectory
end
```

A wide variety of `container`s, `sampler`s, and `controler`s are provided. For the full list, please read the doc.

## Acknowledgement

This async version is mainly inspired by [deepmind/reverb](https://github.com/deepmind/reverb). 
