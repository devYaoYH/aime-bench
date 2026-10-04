# Lightweight AIME 2026 transfer

**Reached 18 distinct verified correct in 88.669s.** One predeclared seed (`20261021`), all 30 AIME 2026 questions, same frozen core v1 and improved-prompt policy as the AIME 2025 five-seed batch. No retuning or replacement seed.

| Measurement | Result |
| --- | ---: |
| Official time to 18 | 88.669s |
| Initialization + official attempt | 154.589s |
| First grader pickup | 6.396s |
| Grader service | 57.002s |
| Later grader idle | 25.271s |
| Grader checks | 19: 18 correct, 1 wrong |
| Generation requests | 46: 30 fresh, 16 continuations |
| Winning continuations | 4 of 18 |
| Fresh / continuation median TTFT | 169.9 / 193.8ms |

The [protocol](../../../configs/experiments/vibe-frozen-core-aime2026-lightweight-v1.json) was committed and pushed before launch. Settings stayed fixed: NVFP4 VibeThinker-3B, Marlin weights, BF16 activations/KV, FlashInfer attention, 95% GPU allocation, 65,536-token total context, 30×1 barrier, 8,192 initial output tokens, up to 16,384 additional tokens per continuation, temperature 0.8/top-p 0.95, and four requests per question including continuations. This attempt started a fresh owned inference server. Only cheap 30×32-token warmup preceded official timing; benchmark mode buffered required evidence and disabled optional profiling.

[Attempt evidence](../../../attempts/20261004T011020.823297Z/README.md) records source `6f253ba25dd45903186da0932e359ae3c6b40510` (clean), core manifest `35d6a06315a0e45441b3ac49bd468a2b94553fb172f378bfebc7ea8cc044070f`, prompt `26b591c39bcf55f4c94f5359dcc90d3c5626a524478eee1c5ce44b5038162364` and the exact dataset/key hashes. The bundled MathArena AIME 2026 snapshot is revision `d2de22f3c656b4f56cf8981212186377d1e23bc3`. Solver requests match its question text with no answer-key fields; grader health and verdicts identify the matching 2026 key. All 30 question records, 18 distinct positive verdicts and first-solved timestamps, the request cap and target timestamp passed audit. All **16 continuation prompts exactly equal the saved predecessor prompt IDs plus output IDs**. This proves exact-ID continuation requests; it does not prove active KV retention.

Fourteen winners arrived in the first coverage round and four from continuations. The last three solved questions were **Q14, Q11, Q9**. Winning candidates were first detected through 12 prose clauses and six closed boxes. The 88.669s target time reconciles with 6.396s before first pickup + 57.002s grader service + 25.271s later idle, within 0.03s. Trace flush took 2.004s and owned service cleanup 4.110s, outside the official target time.

This is a **successful single transfer check**, not a repeated 2026 performance estimate or full 30-question accuracy measurement. Twelve questions were left unverified at the stop target. Dataset, seed and fresh-server lifecycle differ from the AIME 2025 five-seed batch, so timing differences are not isolated causal effects and the viewer keeps separate dataset clusters. The current AIME 2025 headline remains **5/5 reached 18, median 77.277s, range 62.783–82.492s** across declared seeds on one server.

[All results](summary.json), [audited data](analysis.json), [reproduce the audit](reproduce_audit.py), [cleanup evidence](server-evidence.json). Required client evidence is versioned; raw SSE and grader/server logs remain remote. Cleanup confirmed 0 MiB GPU allocation, no compute workers and no 8000/8077 listeners. Coarse server logs sampled at most 7.1% KV occupancy, with no OOM/preemption warning lines. Optional profiling was disabled, so full eviction counters and measured peak VRAM are unavailable.
