# Exact source replay of the best BF16 attempt

Reached 18 in **102.987s**, not reproducing 71.135s. A detached checkout at `bf5788f51b83db12b1607eaccb1434468822ce64` ran the original v1 with fresh managed vLLM/grader services, original profiling and immediate writes. All 30 initial request payloads and the recorded source commit match. The primary checkout stayed unchanged.

Initial median TTFT was 0.207s. The grader completed 19 checks (18 correct, one wrong), with 57.002s service, 40.912s idle and first pickup at 5.072s. Observed official VRAM peak was 78,434 MiB. All 45 generation requests respected the four-request cap. Startup and cleanup are excluded from the first-18 time.

The archived original source passed **108 offline tests** using existing ignored fixtures copied unchanged. Eight driver checks passed before deployment. Imported request/source, first-solved and metadata evidence passed validation. The job exited 0 and cleaned up its owned services. The immutable checkout is retained on callosum with raw traces and logs.

[Controls](config.json), [summary](summary.json), [attempt](../../../attempts/20261003T230915.710605Z/summary.json). This faithfully checks source-version differences but is a single replay, not a reliability estimate.
