# VibeThinker-3B NVFP4 canonical coverage attempt

Attempt `20261003T205350.742196Z` reached **18 verified correct answers in
85.630 seconds** on Callosum's A100 PCIe 80GB. Thirteen solved in the first
round and five through exact-token continuations. The process exited zero;
managed inference and grader services shut down, ports 8000/8077 were released,
and GPU memory returned to zero after cleanup.

Source commit: `90641d10e1319633ec23cc743c148783e5d39703`, with no tracked
remote changes. The run used the committed canonical runner at this revision;
later overhead instrumentation was not present during this attempt.
Model: `r0b0tlab/VibeThinker-3B-NVFP4`, pinned Hugging Face revision
`2fc0013974d1a466e6a5a11839f029d5aff34dc9`. The model and deployed profiles
passed offline integrity verification before launch.

```sh
~/.venvs/vllm/bin/python -u -m src.attempt \
  --model r0b0tlab/VibeThinker-3B-NVFP4 \
  --strategy coverage --parallelism 30 --rollouts 1 \
  --first-pass-max-tokens 8192 --max-tokens 16384 \
  --target-correct 18 --max-attempts-per-question 4 --max-rounds 4 \
  --temperature 0.8 --top-p 0.95 --seed 20261003
```

The profile used 65,536 total context tokens, a 16,384-token generation ceiling,
95% GPU-memory utilization, Marlin NVFP4 weight-only computation with BF16
activations/KV cache, and prefix caching. vLLM selected FlashAttention 2.
Initialization included CUDA warmup and 30 concurrent 32-token inference
warmup requests. The official timer began after warmup.

| Measurement | Result |
| --- | ---: |
| Verified correct / questions attempted | 18 / 30 |
| Official solving time | 85.630 s |
| Initialization plus solving time | 162.337 s |
| Initialization time, excluded from official timer | 76.708 s |
| Questions solved in rounds 1 / 2 | 13 / 5 |
| Generation requests in rounds 1 / 2 | 30 / 17 |
| Maximum requests used by any question | 2 (limit 4) |
| Completed / cancelled / errored streams | 17 / 30 / 0 |
| Observed output token IDs, including cancelled streams | 226,230 |
| Grader queries; correct / wrong | 20; 18 / 2 |
| Global grader toll | 60.000 s |
| Sum of grader queue waits | 63.238 s |
| First-pass median client TTFT | 3.508 s |
| Continuation median client TTFT | 0.192 s |
| Official-phase observed VRAM peak | 78,036.188 MiB (76.21 GiB) |
| All-phase observed VRAM peak | 78,146.188 MiB (76.31 GiB) |
| Median sampled solving GPU utilization | 100% |
| Model-loading memory reported by vLLM | 2.13 GiB |
| KV capacity reported by vLLM | 2,090,608 tokens |
| Largest periodically logged KV occupancy | 6.8% |

All 17 continuation prompt arrays exactly matched their parent prompt IDs plus
all 8,192 generated token IDs. All parents had complete token records. vLLM's
aggregate prefix-cache hit rate rose from 27.2% to 95.9% after continuations;
this aggregate includes warmup traffic and does not establish per-request cache
hit fractions. No waiting inference requests appeared in the periodic logs.
The saved summary preserves first-solved timestamps and grader query IDs.

Compared with the [previous BF16 canonical coverage run](../20261003T200718.717581Z/README.md),
time to 18 verified answers fell from 92.503 to 85.630 seconds: 6.874 seconds,
or 7.43%. Both used the same seed, sampling, question set, concurrency, prompt,
and continuation budgets. Quantization and the VRAM budget changed together
(80% to 95%), so this single comparison does not isolate their individual effects
or establish a repeatable speedup. The remaining twelve questions were stopped
at the target; this is not a full final-accuracy evaluation.

Original configuration, summary, saved requests/responses, token records,
verification events, and GPU samples are versioned beside this report for the
canonical viewer and analysis. Full SSE dumps, grader audits, and logs remain ignored.
Raw traces, token records, GPU samples, grader audit, and service logs remain at
`/home/azureuser/aime-bench/attempts/20261003T205350.742196Z/` on Callosum.
The durable launcher log and zero exit code are recorded at
`/tmp/vibethinker-nvfp4-canonical-20261003.log` and its `.exit` companion.
