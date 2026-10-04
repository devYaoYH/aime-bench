"""Gold-free question API and unchanged serial verification, using localhost only."""
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest

from grader import server_v2
from src.common import ROOT


class GraderQuestionsTests(unittest.TestCase):
    def test_pinned_apex_and_custom_field_rows_do_not_leak_answers(self):
        gold, questions, evidence = server_v2.load_dataset({"dataset": {
            "format": "jsonl", "source": str(ROOT / "grader/data/apex_shortlist.jsonl"),
            "manifest": str(ROOT / "data/source_apex_shortlist.json")}})
        self.assertEqual(len(gold), 47)
        self.assertEqual(evidence["rows"], 47)
        self.assertTrue(all(set(q) == {"problem_idx", "problem"} for q in questions))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dataset.jsonl"
            path.write_text(json.dumps({"number": 91, "text": "Return the requested expression.",
                                        "secret": "PRIVATE_SENTINEL", "solution": "ALSO_PRIVATE"}) + "\n")
            gold, questions, _ = server_v2.load_dataset({"dataset": {"format": "jsonl", "source": str(path),
                "idx_field": "number", "problem_field": "text", "gold_field": "secret"}})
            self.assertEqual(gold[91], "PRIVATE_SENTINEL")
            self.assertNotIn("PRIVATE", json.dumps(questions))

    def test_real_http_questions_are_free_and_verdict_keeps_gold_private(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "key.jsonl"
            path.write_text(json.dumps({"problem_idx": 91, "problem": "Compute a fraction.", "answer": "4/9"}) + "\n")
            gold, questions, evidence = server_v2.load_dataset({"dataset": {"format": "jsonl", "source": str(path)}})
            class Handler(server_v2.Handler):
                pass
            Handler.questions, Handler.provenance = questions, evidence
            Handler.oracle = server_v2.base.Oracle(gold, .01, str(Path(tmp) / "audit.jsonl"), {
                "sha256": evidence["grader_sha256"], "questions_sha256": server_v2.question_digest(questions)})
            server = server_v2.base.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            connection = http.client.HTTPConnection("127.0.0.1", server.server_port, timeout=10)
            try:
                connection.request("GET", "/questions")
                response = connection.getresponse()
                self.assertEqual(response.status, 200)
                body = json.loads(response.read())
                self.assertEqual(body["questions"], questions)
                self.assertEqual(Handler.oracle.seq, 0)
                self.assertTrue(Handler.oracle.q.empty())
                connection.request("POST", "/verify", json.dumps({"index": 91, "candidate": r"\frac{4}{9}"}),
                                   {"Content-Type": "application/json"})
                response = connection.getresponse()
                verdict = json.loads(response.read())
                self.assertTrue(verdict["verdict"])
                self.assertNotIn("gold", verdict)
                self.assertGreaterEqual(verdict["toll_s"], .009)
                self.assertEqual(Handler.oracle.seq, 1)
            finally:
                connection.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()
