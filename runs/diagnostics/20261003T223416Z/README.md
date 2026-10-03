# Callosum slowdown diagnosis — 2026-10-03

The machine has no accumulated inference processes or memory pressure to clear.
Read-only checks around 22:30–22:35 UTC found zero zombies, no vLLM/grader workers
or service listeners, no GPU compute processes, and 0 MiB allocated VRAM. RAM was
216 GiB total with 211 GiB available; the 63 GiB filesystem cache is reclaimable.
There is no swap configured. CPU samples were essentially 100% idle, with no
current CPU/memory/IO pressure and ample disk and shared-memory space.

GPU temperature was 41°C, SM/memory clocks 1410/1512 MHz, power limit 300 W.
There were no reported throttle events, ECC errors, or row-remapping failures.
The accessible kernel journal contained no matching OOM/Xid errors. Idle GPU
utilization reports approximately 17–18% despite no visible compute process;
this observation alone does not establish contention or a hardware fault.
See [health and historical evidence](evidence.json).

## What actually became slower

| Attempt | Initialization | First 18 correct | First-pass median TTFT |
| --- | ---: | ---: | ---: |
| BF16 best, `211557.382358` | 47.60 s | 71.13 s | 0.159 s |
| NVFP4 30×1 profiled, `212812.156610` | 47.20 s | 106.57 s | 0.244 s |
| NVFP4 30×1 benchmark, `215349.326138` | 66.85 s | 106.93 s | 0.209 s |
| BF16 matched benchmark, `220008.957658` | 66.92 s | 113.63 s | 0.174 s |
| NVFP4 dynamic 60, `222009.163270` | 95.45 s | 127.57 s | 3.406 s |

Attempt IDs have the `20261003T` prefix and `Z` suffix. Initialization is outside
the official solving window. Conventional medians are used; the last row means
the 30 initial streams, excluding subsequent fresh samples and continuations.

1. **The newest launch incurred a compilation-cache miss.** Earlier NVFP4 runs
   loaded the same cached AOT artifact and spent 0.20 s in `torch.compile`.
   The expanding-profile run compiled a different artifact and spent **22.62 s**;
   engine initialization rose from 13.49 to 39.18 s. Weight loading stayed at
   0.29 s and full model loading near 3.1 s. Cache-key attribution to any single
   configuration field has not been established. Preserve these caches.
2. **Disabling profiling also removes a GPU driver holder.** GPU persistence is
   off. Two tiny probes measured cold CUDA initialization at 2.835/2.806 s,
   versus 0.264/0.265 s while retaining a temporary parent NVML handle.
   Earlier profiled runners kept their NVML sampler active; benchmark runners
   disable it. This supports repeated driver initialization as a contributor to
   pre-server startup time: the first timestamped API-server log moved from
   about 16 s after initialization began to about 35 s. The probe does not
   account for every second of that difference. See [probe timings](cuda-probes.json).
3. **Solving time has not consistently regressed under matched NVFP4 controls.**
   The 30×1 runs were 106.57 and 106.93 s, with normal TTFT in both. The earlier
   BF16 71-second result did not repeat, but fresh TTFT remained normal; the
   repeat needed 14 continuations and three incorrect checks, whereas the best
   run needed neither. Its answer-arrival tail is not evidence of a memory leak.
4. **The dynamic run is a changed workload.** It admitted up to 60 streams,
   made 140 requests, and grew contexts through continuations. At 60 active
   requests, existing server logs showed approximately 5.8K tokens/s early and
   3.4K tokens/s late as contexts grew. Peak logged KV utilization was only
   17.4%, with no preemption/OOM warning lines. Benchmark mode supplies no
   eviction counter or sampled VRAM peak. Initial TTFT increased substantially;
   its precise cause still needs a matched warmed-server comparison. Warmup had
   four sampler JIT warnings before official timing; none were logged during it.

## Practical next steps

Do not drop Linux caches or reset the GPU: the health checks show nothing to
reclaim, and clearing caches would remove useful model/compilation data. No
processes were killed, caches flushed, services restarted, or driver settings
changed during this diagnosis. The four tiny CUDA probes exited cleanly.

For future comparisons, keep the same server/profile alive with `--reuse-server`
across matched runs, retain compilation caches, and warm the intended batch sizes
before official timing. Compare fixed-length decoding and TTFT separately from
time to 18, which depends on sampled answers and the grader queue. A quiet NVML
handle or GPU persistence could reduce startup costs without polling overhead;
neither has been added to the runner or enabled on the machine here.

This report uses existing attempt evidence plus read-only host diagnostics and
small CUDA initialization probes. It does not prove that GPU decoding throughput
is unchanged under an identical workload. No source or solving policy changed.
