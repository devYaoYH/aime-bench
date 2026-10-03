# BF16 dynamic admission limited to 30 streams

Three benchmark trials reached 18 in 92.013s, 105.395s and 114.298s. None reproduced 71.135s. Each starts the original 30×1 requests, then recycles freed slots to fresh samples or capped continuations, with at most 30 active requests and four generation requests per question. All initial payloads match the original reference.

| Trial | Attempt | First 18 | Wrong checks | Grader idle | Requests |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | [20261003T230345.674426Z](../../../attempts/20261003T230345.674426Z/summary.json) | 92.013s | 2 | 26.964s | 72 |
| 2 | [20261003T230521.104610Z](../../../attempts/20261003T230521.104610Z/summary.json) | 105.395s | 1 | 40.833s | 72 |
| 3 | [20261003T230709.939894Z](../../../attempts/20261003T230709.939894Z/summary.json) | 114.298s | 2 | 50.961s | 72 |

Median 105.395s versus 134.519s in the matched BF16 barrier benchmark repeats. Stochastic output/admission paths differ, so this descriptive comparison does not isolate the scheduling effect. Both use 95%/64K context, temperature 0.8, top-p 0.95 and seed 20261003. Request and stream limits, first-solved records and metadata schemas passed validation. Fourteen relevant offline checks passed before deployment.

[Controls](config.json), [all summaries](summary.json). Raw SSE, service logs and grader audits remain remote. No sampled GPU peaks or eviction counters are available in benchmark mode.
