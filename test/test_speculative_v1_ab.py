import json
from pathlib import Path
import tempfile
import unittest

import yaml
from scripts.speculative_v1_ab import ROOT, MODEL, PROFILES, counters, decode_analysis


class SpeculativeEvidenceTests(unittest.TestCase):
    def test_only_speculation_changes_serving_profile(self):
        profiles = [yaml.safe_load((ROOT/"configs/vllm"/MODEL/PROFILES[a]).read_text()) for a in ("A", "B")]
        spec = json.loads(profiles[1].pop("speculative-config"))
        self.assertEqual(profiles[0], profiles[1])
        self.assertEqual(spec["num_speculative_tokens"], 3)
        self.assertEqual(spec["max_model_len"], 32768)

    def test_counts_token_ids_in_multi_token_chunks_and_audits(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root/"trace/01/rollout-01"
            path.mkdir(parents=True)
            def write(p, data):
                p.write_text(json.dumps(data))
            write(root/"config.json", {"official_started_at_utc": "2026-10-07T00:00:00+00:00"})
            write(path/"telemetry.json", {"started_at_utc": "2026-10-07T00:00:00+00:00",
                  "generated_token_ids_count": 9, "prompt_token_ids_count": 10,
                  "ttft_s": 1, "status": "cancelled"})
            write(path/"tokens.json", {"output_token_ids": list(range(9))})
            events = [{"elapsed_s": t, "data": json.dumps({"choices": [{"token_ids": ids}]})}
                      for t, ids in [(1, [0, 1]), (5, [2, 3, 4]), (10, [5, 6, 7, 8])]]
            (path/"stream.jsonl").write_text("\n".join(json.dumps(e) for e in events))
            result = decode_analysis(root)
            self.assertAlmostEqual(result["requests"][0]["decode_tps"], 7/9)
            self.assertEqual(sum(w["tokens"] for w in result["windows"]), 7)
            self.assertAlmostEqual(result["concurrency_bins"]["0-4"]["per_active_tps"], 7/9)
            self.assertEqual(result["windows"][0]["mean_context_tokens"], 12)
            write(path/"tokens.json", {"output_token_ids": list(range(8))})
            with self.assertRaisesRegex(RuntimeError, "Token evidence mismatch"):
                decode_analysis(root)

    def test_keeps_speculative_counters(self):
        values = counters('vllm:spec_decode_num_draft_tokens_total{model="x"} 100\n'
                          'vllm:spec_decode_num_accepted_tokens_total{model="x"} 40\n')
        self.assertEqual(values["spec_decode_num_accepted_tokens_total"], 40)


if __name__ == "__main__":
    unittest.main()
