# Fresh Callosum slowdown diagnosis

Read-only inspection at 23:25–23:27 UTC found zero zombie processes, no vLLM/grader workers or service listeners, no GPU compute processes, and 0 MiB allocated VRAM. The host has 216 GiB RAM with 211 GiB available, 147 GiB free and about 66 GiB reclaimable buffers/cache. CPU and memory pressure averages were zero; disk has 923 GiB free. No memory flush is supported by these observations.

GPU temperature was 45–47°C, SM/memory clocks 1410/1512 MHz, with no thermal/hardware slowdown, ECC errors or row-remapping failures. As in the earlier diagnosis, NVML reports 17–18% idle GPU utilization despite no visible process or allocation. Its cause is unresolved; this alone does not demonstrate competing inference work.

Shared memory contains one 32-byte semaphore file dating from 20:03 UTC (4 KiB filesystem allocation). No System V semaphore arrays remain. This tiny leftover cannot account for RAM/VRAM pressure and was preserved because its owner/workload was not established.

The latest four warmed NVFP4 30×1 runs took 118.188 / 94.591 / 126.428 / 114.441s to 18 verified answers. Initial median TTFT remained 181 / 229 / 200 / 229ms. Grader service consumed 54–63s, while idle gaps between checks consumed 35–63s, compared with 12.076s in the original 71.135s run. This locates much of the extra time in later correct-candidate arrivals, rather than request startup. The original result still has not repeated.

Latest server logs show about 4.0–4.8K aggregate output tokens/s early at roughly 27–30 streams, and about 1.4–2.0K later with 12–16 streams and longer contexts. These changing workloads are not a controlled throughput comparison. Logged KV usage peaked at 9.6%, with no preemption/eviction/OOM warning lines. Benchmark mode has no sampled VRAM peak or eviction counter, so log absence is not proof that every KV block was retained.

The logged EngineDeadError at 23:22:41 followed explicit SIGTERM and engine teardown after the final trial completed; the controller exited 0. All workers were gone at inspection. This shutdown message does not explain inference latency.

No processes were killed, caches dropped, GPU reset, or persistence/clock settings changed in this diagnosis. Retain useful model/compiler caches. Earlier probes and logs attributed part of slower initialization to cold CUDA initialization and a 22.62s compiler-cache miss; initialization is outside the official time to 18.

[Health and log evidence](evidence.json), [latest seed batch](../../experiments/nvfp4-30x1-seed-comparison-20261003T231357Z/README.md), [earlier startup diagnosis](../20261003T223416Z/README.md), [timing comparison](../../experiments/best-replication-20261003/README.md).
