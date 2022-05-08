# Trajectories

[![Build Status](https://github.com/JuliaReinforcementLearning/Trajectories.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaReinforcementLearning/Trajectories.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/JuliaReinforcementLearning/Trajectories.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/JuliaReinforcementLearning/Trajectories.jl)
[![PkgEval](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/T/Trajectories.svg)](https://JuliaCI.github.io/NanosoldierReports/pkgeval_badges/report.html)

## Design

A typical example of `Trajectory`:

![](https://user-images.githubusercontent.com/5612003/167291629-0e2d4f0f-7c54-460c-a94f-9eb4148cdca0.png)

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
