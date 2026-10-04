# Lightweight AIME 2026 transfer

The [recorded protocol](config.json) declares one seed (`20261021`) on AIME 2026 using `runner_final.run_frozen`, core v1 and `prompt_adherence.json`, with no retuning or replacement seed. Source commit: `6f253ba25dd45903186da0932e359ae3c6b40510`; remote checkout: `/home/azureuser/aime-bench-aime2026-6f253ba`.

Evidence is in [attempt 20261004T011020.823297Z](../../../attempts/20261004T011020.823297Z/README.md). It starts a fresh owned inference server, uses cheap warmup and benchmark mode, admits all 30 questions, and stops at 18 distinct verified correct. Dataset and seed differ from the AIME 2025 improved-prompt batch, so it belongs to a separate dataset cluster. A single transfer attempt cannot establish repeatability or full-dataset accuracy; unavailable timing remains unranked.

The scored run completed with 18 distinct verified correct in **88.669s**, measured after warmup through the eighteenth positive verdict. Settlement was 88.729s. See the attempt's [summary](../../../attempts/20261004T011020.823297Z/summary.json) and saved verification events for the evidence.
