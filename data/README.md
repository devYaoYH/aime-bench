# AIME benchmark data

- AIME 2025: [MathArena/aime_2025](https://huggingface.co/datasets/MathArena/aime_2025), existing development benchmark; provenance in `source.json`.
- AIME 2026: [MathArena/aime_2026](https://huggingface.co/datasets/MathArena/aime_2026), held-out generalization benchmark; provenance in `source_2026.json`.

Each year has 30 ordered problems: indices 1–15 are AIME I, 16–30 are AIME II.
`aime_<year>_problems.jsonl` contains the MathArena transcription and integer gold
answers. The matching `../grader/data/aime_<year>.jsonl` uses string gold answers
for the vendored grader. Solver loaders return only indices and problem text.

The 2026 data is downloaded from Hugging Face commit
`d2de22f3c656b4f56cf8981212186377d1e23bc3`, not transcribed from search snippets.
The manifest records the pinned URL, downloaded Parquet hash, prompt/key hashes,
retrieval timestamp, and upstream license. No wording or answer fixes are applied.

MathArena documents [different US/international AIME II variants](https://huggingface.co/datasets/MathArena/aime_2026/discussions/2).
Its #1 (index 16) has answer 178 and #10 (index 25) has answer 850. Preserve those
statements and answers together when comparing against external results.

MathArena distributes these datasets under **CC BY-NC-SA 4.0**. Original contest
problems are credited to the Mathematical Association of America (MAA AMC).
The upstream `train` split name describes packaging; use the recorded benchmark
role to distinguish development runs from generalization tests. Downloading this
year establishes a separate local test set; it does not establish that a model
has never seen the problems during pretraining.

Refresh with `python -m src.fetch_dataset --year <2025|2026> --revision <commit>`.
This replaces that year's prompt/key files and manifest. It leaves the other year
untouched. `pyarrow` is needed only for the downloader.
