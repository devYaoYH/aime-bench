# Time allocation and unattended compute

This confirmed post-freeze task only. Earlier human focused hours were not recorded and are not reconstructed. Activity intervals are wall-clock categories, not measured human attention.

| UTC start | UTC end | Category | Wall time |
| --- | --- | --- | --- |
| 2026-10-04T10:31:05Z | 2026-10-04T10:41:56Z | focused agent-assisted development | 651.000s |
| 2026-10-04T10:41:56Z | 2026-10-04T11:29:03.692+00:00 | unattended background compute | 2827.692s |
| 2026-10-04T10:41:56Z | 2026-10-04T11:40:21+00:00 | analysis/reporting and monitoring wall time, including waits | 3505.000s |

Analysis and background compute overlap. Do not sum them as focused hours. Earlier human focused hours are unverified. The 261-test offline suite took 12.768s within the development interval.

## GPU jobs

| Job | UTC start | UTC end | Background wall time |
| --- | --- | --- | --- |
| Task A, unattended background compute | 2026-10-04T10:41:56.333+00:00 | 2026-10-04T11:21:42.815+00:00 | 2386.482s |
| Seed 20261011, unattended background compute | 2026-10-04T10:42:52.439+00:00 | 2026-10-04T10:51:06.925+00:00 | 494.486s |
| Seed 20261012, unattended background compute | 2026-10-04T10:51:06.926+00:00 | 2026-10-04T10:58:42.046+00:00 | 455.120s |
| Seed 20261013, unattended background compute | 2026-10-04T10:58:42.048+00:00 | 2026-10-04T11:06:28.185+00:00 | 466.137s |
| Seed 20261014, unattended background compute | 2026-10-04T11:06:28.187+00:00 | 2026-10-04T11:14:12.235+00:00 | 464.048s |
| Seed 20261015, unattended background compute | 2026-10-04T11:14:12.237+00:00 | 2026-10-04T11:21:38.702+00:00 | 446.465s |
| Task B, unattended background compute | 2026-10-04T11:21:42.821+00:00 | 2026-10-04T11:29:03.692+00:00 | 440.871s |

Task intervals include service setup and cleanup. Trial intervals include fresh grader setup, cache reset, cheap warmup and final trace flush. Official clocks exclude those costs. Task A records inference launch separately (56.107s to ready). Its 900s deadline was armed only at official start; every seed ended naturally before that deadline.

The background batch ran sequentially from 10:41:56.333 to 11:29:03.692 UTC (47m7.359s). Task B has one declared seed; no replacement or second accuracy seed was run. Reporting-work end is the last logged documentation/QA checkpoint, not a measurement of human attention.
