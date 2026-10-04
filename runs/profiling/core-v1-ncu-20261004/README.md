# Aborted initial Nsight instrumentation trial

This **unranked, interrupted** core v1 attempt is retained to account for the
profiling retry. It did not reach the target and provides no decode roofline.
Attempt: [20261004T080910.736443Z](../../../attempts/20261004T080910.736443Z/README.md).

The first collection included the full SpeedOfLight and hierarchical Tensor
roofline sections, with auxiliary kernels as well as whole graphs. Nsight backed
up approximately 80 GB of allocated device memory into host RAM during replay.
The profiler control request exceeded its 180s timeout while collecting auxiliary
kernels, before a decode graph was reached. The controller interrupted the owned
runner, flushed its partial traces, and stopped the owned services.

The [revised capture](../core-v1-ncu-20261004-single-pass/README.md) restricted
collection to whole decode graphs and four essential counters, validated in a
single-pass synthetic CUDA graph probe. It completed one full diagnostic attempt.
The model profile, frozen core and solving hyperparameters were preserved.

[Configuration](config.json) and [failed controller outcome](summary.json) are
versioned. Full logs and the partial binary profile remain on the node in
`/home/azureuser/aime-bench-ncu-631316c7/`. Source commit and exact commands are
recorded in the configuration.
