# Post-freeze report revision

## Structure and evidence

- Lead with the measured five-seed core v1 result, then explain the runner and
  its latency objective before discussing optimization choices.
- Keep the historical 59.316-second draw secondary, with the failed validation
  gate visible. Distinguish that earlier batch's settling trial from the final
  five-seed batch, which retained every declared outcome.
- Move detailed warmup, source-control, CPU-method, backend and reference lists
  into the evidence packet. Preserve every historical packet section and figure.
- Add full-coverage milestone and complete-baseline evidence after collection.
  Keep these separate from historical stop-at-18 and cancellation-enabled runs.
- Add the recorded constant-concurrency refill outcomes and the selected-core
  exact-token tail analysis. Treat proposed probing as future work.

## Corrections and scope

- Total grader service already includes incorrect checks. The latency
  decomposition separates correct and incorrect service instead of adding
  wrong-check cost to total service a second time.
- The 78.601-second AIME 2024-warmed median is a superseded preset, rather than a
  second current headline. The measured improved-prompt headline is 77.277s.
- The 6–27-second semantic-result rechecking observations belong to the older
  AIME 2024-warmed cohort. Selected-core token replay has its own provenance.
- Historical experiment counts remain fixed snapshots. Newly collected runs
  are reported separately rather than silently extending their figures.
- Total historical human focused hours were not recorded and could not be
  verified. The new time log records actual task activity intervals separately
  from unattended compute and does not infer human attention.

## Cuts

Repeated qualifications and redundant main-report reference lists move to the
packet and citation docket. No experimental evidence is removed from the packet.
Final validation: three report pages and 15 evidence pages at the original
10.1pt body font. All pages were visually reviewed; E0–E9 and Appendix A retain
their historical page numbers. All 26 report packet citations and 152 local
Markdown evidence links resolve. The renderer now exports actual named PDF
destinations for external citation links, rather than only outline bookmarks.

Five extended outcomes are retained: median time to 18 is 78.305s, range
63.908–135.071s; every seed reached 26 and none reached 28. The complete BF16
baseline supplies all 120 sample mappings, 47.50% pass@1, 18/30 pass@4 and
plurality voting, 13/30 strict three-of-four voting, and 52.50% capped/no-answer
rates. Peak sampled baseline VRAM was 76.719GiB. These measurements are
separate from the original headline cohort.

The pre-execution suite passed 261 offline tests. Independent post-run analysis
verified the frozen source controls, 150 same-seed initial payloads, 126 exact-ID
continuations and all 120 baseline terminal records/verdict mappings. Metadata
validation passed for the imported evidence. No historical human focused-hours
estimate or current-prompt semantic rechecking delay could be verified; both
remain explicitly scoped.
