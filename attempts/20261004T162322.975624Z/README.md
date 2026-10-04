# Core v2.1 benchmark: AIME 2025

Reached 18 verified questions in **82.444820s** (official settlement 82.496534s),
with 20 completed grader checks: 18 correct and 2 wrong. One seed `20261011`.
Measured source `e5641930`, manifest
`7ca23622428105d4d1de78fdd4df0250e07aab3a28c0a842657919903c117948`.

NVFP4 VibeThinker-3B, configured 95% GPU memory, unchanged v2 prompt, 30×1 barrier,
8K initial / 16K additional continuation output, four generation requests per
question, 64K context. Owned inference and grader services; cheap 30×32 warmup.
Setup/warmup and final buffered trace flush are outside official solving time.

291 proposals: 21 unique candidates enqueued, 157 placeholders rejected,
113 expression duplicates suppressed. CPU validation recorded 0.166055s of
worker process time. There were 265 cache hits and 26 misses; uncached median/max
wall time was 6.370/32.537ms. Wall timers include IPC and queue waits; these sums
are not critical-path overhead estimates. No normalization fallback occurred.

45 generation requests (30 fresh, 15 continuations) all saved complete exact
prompt/output IDs, including 24 cancelled streams. Fresh/continuation median
TTFT was 0.196922/0.173856s. Required verification, validation, generation and
first-solved timing evidence is versioned here. Benchmark mode disables optional
host/GPU/engine profiling, so measured VRAM/KV series are unavailable. Raw SSE,
grader audits and service logs remain in the pinned remote worktree
`/home/azureuser/aime-bench-v2_1-e5641930`.

Historical same-seed v2 control `20261004T013249.730690Z` reached 18 in 136.240866s
with 37 completed checks and 19 wrong. The recorded model/profile, prompt,
sampling and scheduling controls match; execution time/core/parser dependencies
differ. This single pair does not establish repeatability or isolate a causal
latency effect. See the [two-dataset timing report](../../runs/experiments/core-v2_1-benchmarks-20261004T161700Z/README.md).
