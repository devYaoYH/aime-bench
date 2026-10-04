# Core v2.1 benchmark: Apex shortlist

**13/47 verified questions at the 900-second official deadline; target 18 unmet.**
This is a preserved interrupted trial, not an exhausted-budget result and not a
ranked time-to-18 measurement. Official settlement took 900.057531s. The client
completed 35 checks: 13 correct, 22 wrong; one additional candidate was enqueued.
Seed `20261011`, measured source `e5641930`, frozen v2.1 manifest
`7ca23622428105d4d1de78fdd4df0250e07aab3a28c0a842657919903c117948`.

NVFP4 VibeThinker-3B, configured 95% GPU memory, unchanged v2 prompt, up to
30×1 question concurrency with barrier retries, 8K initial / 16K additional
continuation output, four generation requests per question and 64K context.
All 47 question statements were fetched from the grader; Apex #25/#26 overlap
AIME 2025. Owned inference/grader services; cheap 30×32 warmup. Setup/warmup and
final buffered trace flush are outside official solving time.

317 proposals: 36 unique candidates enqueued, 148 rejected (141 placeholders,
7 invalid LaTeX), 133 equivalent repeats suppressed. Worker CPU validation
cost was 0.809030s; 273 cache hits / 44 misses. Uncached median/max wall time:
7.382/264.167ms. Seven syntax-valid normalization failures fell back to raw
keys and remained eligible for grading. Required traces do not establish that
all syntax-rejected candidates were mathematically incorrect.

132 generation requests (52 fresh, 80 continuation) all saved complete exact
prompt/output token IDs. Fresh/continuation median TTFT was 0.176314/0.333121s.
The completed grader jobs occupied 105.004s, with 674.834s idle between those
jobs; the first job began at 82.481s. This is the observed completed-job timeline,
not an account of additional service that cancelled jobs might incur.

There is no matched Apex control in this batch. Benchmark mode omits optional
host/GPU/engine profiles, so measured VRAM/KV series and extraction/IO overhead
are unavailable. Raw SSE, grader audits and service logs remain at
`/home/azureuser/aime-bench-v2_1-e5641930/attempts/20261004T162558.973938Z`.
See the [two-dataset timing report](../../runs/experiments/core-v2_1-benchmarks-20261004T161700Z/README.md).
