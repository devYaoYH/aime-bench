# Speedrun Grader

A minimal HTTP **answer-checking oracle** for math speedruns. A solver asks
*"is this candidate answer correct for problem N?"* and gets back **yes/no** — but
**every verification costs `cost_c` seconds (default 3 s)**, and the gate is **global
across every client and process**. Grading itself is instant (sympy math
equivalence); the delay is the deliberate **cost** of a verification — as if a human
grader needs `cost_c` to check each answer.

## How it works
- One process (`server.py`) loads the dataset's gold answers and serves HTTP.
- It runs a **single global FIFO queue with one worker**. Each `/verify` request is
  queued; the worker answers them one at a time, **≥ `cost_c` seconds apart**, in
  arrival order, each tagged with the time it was answered (`answered_at`) and
  appended to `logs/queries.jsonl`.
- It uses a **toll model**: each query is charged `cost_c` from when the worker picks
  it up, so **every** verification — including the first — costs `cost_c`.
- One queue + one clock in one process ⇒ the limit is **global**: many clients across
  many processes still share *one answer per `cost_c`*.
- Verdicts use the MathArena parser, so `0.5` == `\frac{1}{2}`, `\boxed{204}` == `204`,
  list/tuple/interval answers, etc. Gold answers never leave the server — clients only
  ever get one bit.

## Setup (self-contained)
The tool has its **own venv and pinned deps** — it does not depend on any other
project's environment.
```bash
cd speedrun-grader
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
PY=.venv/bin/python          # used in the commands below
```
(`datasets` is only needed for the HF-hosted AIME/Apex sets; for a purely local
jsonl/csv dataset you can drop it from `requirements.txt`.)

## Start it
The default `config.yaml` points at the **bundled** `data/aime_2025.jsonl` (30 problems),
so it runs out of the box with no network:
```bash
$PY server.py
# loaded 30 problems (index 1..30); cost_c=3.0s
# grader listening on http://127.0.0.1:8077  (POST /verify, GET /health)
```
The problems to solve are in `data/aime_2025.jsonl` (each row: `problem_idx`, `problem`,
`answer`); the grader only reads `problem_idx` + `answer`.

## Query it from another Python program
It's just an HTTP POST. Two runnable examples are included (start the server first):
- `example_usage.py` — single verify + a 5-thread concurrency demo.
- `example_usage_async.py` — an **asyncio** client that fires a mix of correct and
  incorrect answers and prints a full client-side input/output log with time-tags.

Minimal form:
```python
import json, urllib.request
def verify(index, candidate, base="http://127.0.0.1:8077"):
    body = json.dumps({"index": index, "candidate": candidate}).encode()
    req = urllib.request.Request(base + "/verify", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:   # long: may queue
        return json.loads(r.read())

r = verify(1, "204")
print(r["verdict"], r["answered_at"])   # True/False + time-tag
```
Or from the shell:
```bash
curl -s localhost:8077/verify -H 'Content-Type: application/json' \
     -d '{"index": 1, "candidate": "204"}'
```

## Concurrent queries
Fire as many as you like from as many threads/processes as you like — the grader
**serializes** them through its single gate. With `cost_c=5` and 5 simultaneous
queries, each costs `cost_c`, so the answers land at 5, 10, 15, 20, 25 s (≈25 s total),
in arrival order, each time-tagged. Run `$PY example_usage.py` (threads) or
`$PY example_usage_async.py` (asyncio, mixed correct/incorrect) to see it. Pace your
verifications: every one costs `cost_c` of wall-clock against the others.

## Config (`config.yaml`)
| key | meaning |
|---|---|
| `dataset.source` | local file path (default `data/aime_2025.jsonl`, resolved against this dir) or an HF dataset name |
| `dataset.format` | `jsonl` \| `csv` \| `parquet` \| `hf` |
| `dataset.idx_field` / `gold_field` | column names for the index and the gold answer |
| `cost_c` | seconds each verification costs (toll, charged from worker pickup) |
| `host` / `port` | bind address |
| `audit_log` | append-only JSONL of every answered query |

## Switch dataset
Same schema, so just change `dataset.source` (+ `format`) in `config.yaml`:
- **AIME-2025 (default):** the bundled `data/aime_2025.jsonl` — `format: jsonl`,
  30 problems, indices 1..30. No network.
- **Apex-Shortlist:** `source: MathArena/apex-shortlist`, `format: hf` — 47 problems
  (needs network + `datasets`). To bundle it locally too, dump it to a jsonl with the
  same `problem_idx`/`answer` columns and point `source` at the file.

Then restart the server. Confirm with `curl -s localhost:8077/health`.

## Audit log
Every answered query appends one self-describing JSON line to `logs/queries.jsonl`:

| field | meaning |
|---|---|
| `query_id` | unique id for this query (server-generated, or echoed from the client) |
| `seq` | global answer order (1, 2, 3, …) |
| `agent_id` | optional caller label (else `null`) |
| `index` | problem index queried |
| `candidate` | the answer string the client submitted |
| `gold` | the correct answer — **log only**, never returned to clients |
| `verdict` | `true`/`false` |
| `queue_ahead_at_submit` | jobs already waiting ahead of this one when it arrived |
| `submitted_at` / `picked_at` / `answered_at` | time-tags (submitted → worker started → answered) |
| `queue_wait_s` | wait before grading started |
| `toll_s` | the cost actually paid (≈ `cost_c`) |
| `latency_s` | total felt time (`queue_wait_s` + `toll_s`) |

The HTTP `/verify` response is this same record **minus `gold`**.

## Reset

- **Run state (`seq` counter + queue) is in memory only.** Just **restart the server**
  (Ctrl-C, then `$PY server.py`) — it comes back fresh, `seq` from 1, with an empty
  queue. The toll gate has no persisted state, so a restart fully clears it; nothing
  carries over between runs.

- **Clear / archive the audit log** (the only thing written to disk). Stop the server
  first, then:
  ```bash
  : > logs/queries.jsonl                                   # truncate to empty
  # or keep a copy:
  mv logs/queries.jsonl logs/queries.$(date +%Y%m%d_%H%M%S).jsonl
  ```
  The server recreates `logs/queries.jsonl` on the next query.

- **Full clean reset** (e.g. a corrupted environment):
  ```bash
  rm -rf .venv __pycache__ grader_core/__pycache__ logs/queries.jsonl
  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
  ```
  `config.yaml` and the bundled `data/aime_2025.jsonl` are static — nothing to reset
  there.


## AIME 2026

The pinned 30-problem MathArena 2026 key is bundled in `data/aime_2026.jsonl`.
For a standalone oracle, use `GRADER_CONFIG=config_2026.yaml python server.py`
from this directory. Managed runners select it with `--benchmark-year 2026`.
`/health`, verification responses, and audit records expose dataset configuration
and the actual key-file SHA256; gold answers remain absent from HTTP responses.
The default `config.yaml` still selects 2025. See [source provenance](../data/README.md).
