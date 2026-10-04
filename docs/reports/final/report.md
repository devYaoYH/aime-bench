# Faster verified math answers

**Callosum speedrun report | 4 October 2026**

## 1. Objective, result and runner

The objective is to reach **18 distinct verified-correct answers out of 30 AIME 2025 questions** as quickly as possible on one A100 PCIe 80GB. The original stop-at-18 batch of frozen core v1 with the improved prompt reached 18 in **5/5 declared trials**, with median **77.277s** and range **62.783-82.492s**. The fastest historical draw was 59.316s, from a separate validation batch whose repeatability gate failed. [E5](output/pdf/callosum-evidence-packet.pdf#nameddest=e5), [E8](output/pdf/callosum-evidence-packet.pdf#nameddest=e8), [E9](output/pdf/callosum-evidence-packet.pdf#nameddest=e9)

The grader serves one FIFO queue and charges three seconds for every check. Eighteen correct checks therefore require at least **54 seconds**. Diagnose elapsed time as first grader pickup, correct-check service, wrong-check service and later idle gaps. For BF16 30 x 1, 5.055s before pickup + 54.002s service + 12.076s idle reconstructs about 71.135s; this run had no wrong checks. Wrong service is already included when using total service. [E1](output/pdf/callosum-evidence-packet.pdf#nameddest=e1)

The useful observation is that the model can derive an answer before finishing its checks. All 30 questions start one streaming rollout, initially capped at 8,192 output tokens. The prompt asks for a prospective integer answer immediately. A static CPU parser recognizes complete boxes, answer lines and supported prose clauses as chunks arrive. Each question deduplicates integers and keeps at most one grader request pending; other questions can queue their own checks. Generation continues during verification. A correct verdict cancels that question's remaining generation, and the eighteenth distinct positive verdict stops the attempt. [E0](output/pdf/callosum-evidence-packet.pdf#nameddest=e0)

Barrier scheduling completes a coverage round before continuing unsolved questions. Capped trajectories resume their exact token IDs; later requests allow **16,384 additional output tokens**, clipped to total context. Natural completion or exhausted context can start a fresh sample. Four requests per question include continuations. Serving uses NVFP4 VibeThinker-3B, Marlin weights, BF16 activations/KV, FlashInfer attention, 95% allocation and 65,536-token total context. [E0](output/pdf/callosum-evidence-packet.pdf#nameddest=e0), [E10](output/pdf/callosum-evidence-packet.pdf#nameddest=e10)

Timing excludes server launch, the cheap 30-stream x 32-token warmup, cleanup and buffered trace flush. Benchmark mode keeps required evidence in RAM and disables optional profiling. The final five-seed batch scored every declared trial; an earlier FlashInfer validation had a separate 136.048s settling trial. [E3](output/pdf/callosum-evidence-packet.pdf#nameddest=e3), [E5](output/pdf/callosum-evidence-packet.pdf#nameddest=e5)

Start with the [repository quickstart](../../../README.md#run-the-measured-core-v1). The measured entrypoint is `python -m runner_final.run_frozen --preset runner_final/presets/prompt_adherence.json`. Its manifest rejects core drift; change prompts and presets explicitly, and version new policies separately.

<!-- pagebreak -->

## 2. What worked and what did not

| Measured comparison | Time to 18 | Decision |
| --- | --- | --- |
| BF16 final-only pass@4 vs coverage/continuation | 336.497s vs 92.428s | The complete strategy was 3.64x faster; controls differed. E0 |
| BF16 initial 30 x 4 | 114.615s | Wider fan-out slowed individual trajectories. E1/E2 |
| NVFP4 dynamic60 | 127.565s | More active samples did not reduce the tail. E4 |
| Eager 30-slot refill, five trials | 77.352s median | It did not improve core v1's 77.277s median. E13 |

**Concurrency trades throughput for answer latency.** Per-active-request decode fell from 115.5 to 94.6 to 69.7 tok/s at 30 x 1, 2 and 4. The 30 x 4 run accumulated 46.279s of grader idle. Zero measured preemptions and at most 20.04% KV occupancy did not support eviction as the explanation in that sweep. [E1](output/pdf/callosum-evidence-packet.pdf#nameddest=e1), [E2](output/pdf/callosum-evidence-packet.pdf#nameddest=e2)

Dynamic60 accumulated 61.573s of idle. The earlier BF16 observations showed about 115 tok/s at 30 active requests versus 77 tok/s at about 53. Eager refill also used **71.8% more requests** across five trials without improving the median. These results favor keeping individual trajectories fast before adding coverage. [E2](output/pdf/callosum-evidence-packet.pdf#nameddest=e2), [E4](output/pdf/callosum-evidence-packet.pdf#nameddest=e4), [E13](output/pdf/callosum-evidence-packet.pdf#nameddest=e13)

**Quantization improved observed decode.** NVFP4/Marlin increased 112.5 to 136.6 tok/s (+21%) with FLASH_ATTN fixed. TTFT slowed from 163 to 205ms. First submission fell from 6.71 to 4.48s, but p=0.056 did not establish improvement. FlashInfer produced the fastest observed trials; its earlier validation gate still failed. [E5](output/pdf/callosum-evidence-packet.pdf#nameddest=e5), [E7](output/pdf/callosum-evidence-packet.pdf#nameddest=e7)

**Keep extraction and verification simple.** CPU model extraction was slow, model verification accepted five of seven wrong answers, and confidence pruning rejected eventual successes. In hosted-model voting evidence, strict majority covered 15/30 versus pass@8's 22/30, while waiting for votes adds latency. These findings informed the design rather than establishing local VibeThinker gains. [A1](output/pdf/callosum-evidence-packet.pdf#nameddest=a1)

**Full baseline accuracy is now measured.** At BF16, 16K total context and 95% allocation, all **120 samples** finished: pass@1 **57/120 (47.50%)**, pass@4 **18/30**, and unique-plurality voting **18/30** (ties/no votes fail). A strict three-of-four vote covers **13/30**. All 57 naturally completed samples were correct. Token-capped and no-answer rates were **52.50% / 52.50%**. Generation took 385.848s, all grading 385.857s, and 18 correct arrived at 366.706s. This is one seed with cancellation disabled, distinct from the historical 80% timing baseline. [E12](output/pdf/callosum-evidence-packet.pdf#nameddest=e12)

<!-- pagebreak -->

## 3. The remaining tail, next steps and handoff

![Five declared-seed extended milestone medians and observed ranges; labels show seeds reaching each later milestone](../../../results/post_freeze/measurements-v1-20261004T104200Z/extended-milestones.png)

The historical core banked 14 answers at a median **49.086s**, only 7.086s above the 42s floor. Its next four increased the milestone median by **28.191s**. The late-answer tail is the main opportunity. The new 135.071s outlier spent 75.037s idle with zero wrong checks. [E11](output/pdf/callosum-evidence-packet.pdf#nameddest=e11) [E8](output/pdf/callosum-evidence-packet.pdf#nameddest=e8)

The new extended trials reached 18 in **5/5**, median **78.305s**, range **63.908-135.071s**. At 24 correct, 5/5 reached the milestone at median 252.871s; at 26, 5/5 reached it at median 333.859s. The 28/30 milestones were reached by 0/5 and 0/5. These are sequential same-seed measurements, not an interleaved comparison. [E10](output/pdf/callosum-evidence-packet.pdf#nameddest=e10), [E11](output/pdf/callosum-evidence-packet.pdf#nameddest=e11)

Next, measure the gap from a requested result's first appearance to parser detection and submission, separating reasoning, format delay and grader queueing. An earlier warmup cohort spent 6-27s rechecking after results appeared. When those delays recur and the grader is idle, test an end-of-thinking probe on a cached prefix, counting it against the request cap. Hosted Qwen Python raised final-only coverage from 15/30 to 19/30; local VibeThinker tool use remains unvalidated. [E13](output/pdf/callosum-evidence-packet.pdf#nameddest=e13), [A1](output/pdf/callosum-evidence-packet.pdf#nameddest=a1)

AIME 2026 reached 18 in **88.669s**, with 19 checks and one wrong, on one seed without retuning. The selected integer parser fits AIME; broader answer domains need the separate general-answer extension. This new batch's first launch-through-18 time was **135.249s**, including setup and warmup, from one cold observation. [E5](output/pdf/callosum-evidence-packet.pdf#nameddest=e5), [E10](output/pdf/callosum-evidence-packet.pdf#nameddest=e10)

Preserve source/preset hashes and every declared outcome. Compare the eighteenth positive verdict, wrong checks, idle time and request usage, while keeping setup and cleanup separate. The [evidence packet](output/pdf/callosum-evidence-packet.pdf) retains methods and raw-record routes; [time allocation](../../../results/post_freeze/measurements-v1-20261004T104200Z/time-allocation.md) separates logged work from unattended compute. Earlier human focused hours were not recorded.
