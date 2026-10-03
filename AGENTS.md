# Working on aime-bench

## Local development and remote experiments

- Write and edit source files in the local checkout. Run relevant offline tests
  locally before committing; run the full suite when changing shared behavior.
- Commit and push the tested changes to this repository. Then use `ssh callosum`,
  change to `/home/azureuser/aime-bench`, and run `git pull --ff-only` on the
  corresponding branch before executing experiments. Check both checkout states
  and commit IDs; preserve unrelated local and remote changes. Do not copy edited
  source files directly to the remote or develop there.
- The remote is an A100 PCIe 80GB machine. Inspect `nvidia-smi`, active processes,
  and service health before launching workloads. Do not terminate an unrelated
  experiment or inference server. Use an explicitly requested model or document
  the choice for a smoke experiment.
- Remote model weights and launch configurations are in
  `/home/azureuser/models/<organization>/<model>/vllm.yaml`. Read that profile and
  `~/models/README.md` rather than inventing launch settings. vLLM uses
  `~/.venvs/vllm/bin/python` and `~/.venvs/vllm/bin/vllm`.
- Use `python -m src.attempt --model <organization>/<model>` for canonical
  attempts. It manages warmup, vLLM, the vendored grader, traces, and telemetry.
  `--reuse-server` deliberately attaches to an idle matching server; it must not
  stop that server. The default starts its own server and rejects occupied ports.
- Use `tmux` or another durable job session for long remote runs. Save command,
  Git commit, model config, timestamps, and results. Inspect progress and errors;
  a launched process alone does not establish success.
- Attempt traces and GPU samples remain on the remote under `attempts/` and are
  ignored by Git. Version small configs, summaries, and reports when useful.
  Never commit credentials, weights, virtual environments, or raw traces.
- To bring back results, copy only experiment artifacts (for example with `scp`),
  review them locally, and commit/push reports from the local checkout. Do not
  commit or push code from the remote machine.

## Checks

Run from this repository root:

```sh
.venv/bin/python -m unittest discover -s test -v
```

Tests are offline. Some legacy tests need local ignored traces or macOS sandbox
support; report those limitations rather than modifying fixtures to hide them.
