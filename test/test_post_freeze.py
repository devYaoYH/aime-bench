"""Full-batch completion, oracle-only accuracy and official deadline safeguards."""
import ast
import asyncio
import json
from pathlib import Path
import tempfile
import unittest

import httpx

from src.attempt_runners import naive_pass4_full_v1 as full, naive_pass4_v1 as original
from src.experiments.post_freeze import with_official_deadline, plan
from src.experiments.post_freeze_analysis import baseline_accuracy, aggregate_milestones
from src.common import atomic_json, utc_now
from runner_final.integrity import verify_core
from src.common import ROOT
from test.test_naive_pass4_v1 import FakeGPU, Stream, chunk


class FullBaselineTests(unittest.IsolatedAsyncioTestCase):
    async def test_correct_verdict_leaves_siblings_running_and_deduplicates(self):
        submitted, callbacks, streams = [], [], []
        outputs = [(r"\boxed{70}", "stop", .001),
                   (r"\boxed{69}", "stop", .01),
                   (r"\boxed{070}", "stop", .02),
                   (r"\boxed{70}", "length", .03)]
        async def handler(request):
            if request.url.path == "/verify":
                candidate = json.loads(request.content)["candidate"]
                submitted.append(candidate)
                return httpx.Response(200, json={"verdict": candidate == "70", "query_id": candidate})
            text, finish, delay = outputs[len(streams)]
            stream = Stream([chunk(text, finish=finish), b"data: [DONE]\n\n"], delay=delay)
            streams.append(stream)
            return httpx.Response(200, stream=stream)
        args = full.parse_args(["--model", "WeiboAI/VibeThinker-3B"])
        def solved(event):
            callbacks.append(event)
            self.assertFalse(all(stream.closed for stream in streams))
        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                row = await full.run_question({"problem_idx":1,"problem":"Compute.","prompt_tokens":20},
                                              args,client,Path(tmp),FakeGPU(),on_solved=solved)
            accuracy = baseline_accuracy(tmp,4,4)
        self.assertEqual(submitted,["70","69"])
        self.assertEqual(len(callbacks),1)
        self.assertTrue(all(r["status"]=="completed" for r in row["rollouts"]))
        self.assertTrue(accuracy["valid"])
        self.assertEqual((accuracy["correct_samples"],accuracy["unique_grader_checks"]),(2,2))
        self.assertEqual((accuracy["pass1"],accuracy["no_answer_rate"],accuracy["token_capped_rate"]),(.5,.25,.25))

    async def test_all_120_samples_complete_beyond_18_solved(self):
        streams, checks = [], []
        async def handler(request):
            if request.url.path == "/verify":
                checks.append(json.loads(request.content))
                return httpx.Response(200,json={"verdict":True})
            stream=Stream([chunk(r"\boxed{70}",finish="stop"),b"data: [DONE]\n\n"],delay=.003)
            streams.append(stream)
            return httpx.Response(200,stream=stream)
        args=full.parse_args(["--model","WeiboAI/VibeThinker-3B","--target-correct","18"])
        problems=[{"problem_idx":i,"problem":f"q{i}","prompt_tokens":20} for i in range(1,31)]
        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                rows=await full.run_baseline(problems,args,client,Path(tmp),FakeGPU(),0)
            accuracy=baseline_accuracy(tmp)
        self.assertEqual((len(streams),len(checks),len(rows)),(120,30,30))
        self.assertTrue(all(s.closed for s in streams))
        self.assertTrue(all(r["status"]=="completed" for q in rows for r in q["rollouts"]))
        self.assertEqual((accuracy["pass1"],accuracy["pass4_correct_questions"]),(1,30))


class DeadlineTests(unittest.IsolatedAsyncioTestCase):
    async def test_deadline_excludes_initialization_and_flushes_partial_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempts=Path(tmp)
            async def run(args):
                output=attempts/"trial";output.mkdir()
                await asyncio.sleep(.05)
                atomic_json(output/"config.json",{"official_started_at_utc":utc_now()})
                try:
                    await asyncio.Event().wait()
                finally:
                    atomic_json(output/"summary.json",{"status":"interrupted"})
            state=await with_official_deadline(run,None,attempts,.03)
            self.assertTrue(state["deadline_reached"])
            self.assertEqual(state["attempt_id"],"trial")
            self.assertEqual(json.loads((attempts/"trial/summary.json").read_text())["status"],"interrupted")

    async def test_external_interruption_retains_attempt_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempts = Path(tmp)
            state = {}
            async def run(args):
                output = attempts / "interrupted"; output.mkdir()
                try:
                    await asyncio.Event().wait()
                finally:
                    atomic_json(output / "summary.json", {"status": "interrupted"})
            task = asyncio.create_task(with_official_deadline(run, None, attempts, 900, state=state))
            await asyncio.sleep(.01)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
            self.assertEqual(state["attempt_id"], "interrupted")
            self.assertFalse(state["deadline_reached"])

    async def test_natural_completion_and_failure_are_retained(self):
        with tempfile.TemporaryDirectory() as tmp:
            attempts=Path(tmp)
            async def run(args):
                output=attempts/"done";output.mkdir();return output
            state=await with_official_deadline(run,None,attempts,900)
            self.assertFalse(state["deadline_reached"])
            self.assertEqual(state["attempt_id"],"done")
            async def failed(args):
                (attempts/"failed").mkdir()
                raise RuntimeError("test failure")
            state=await with_official_deadline(failed,None,attempts,900)
            self.assertEqual(state["attempt_id"],"failed")
            self.assertIn("test failure",state["error"])


class EvidenceTests(unittest.TestCase):
    def test_original_extractor_and_prompt_are_preserved(self):
        def definition(module,name):
            return ast.dump(next(n for n in ast.parse(Path(module.__file__).read_text()).body if getattr(n,"name",None)==name),include_attributes=False)
        self.assertEqual(full.PROMPT,original.PROMPT)
        self.assertEqual(definition(full,"final_answer"),definition(original,"final_answer"))
        self.assertEqual(plan()["task_a"]["core_manifest_sha256"],verify_core(ROOT))
        self.assertEqual(plan()["task_b"]["gpu_memory_utilization"],.95)

    def test_unreached_milestones_are_missing_and_reach_counts_are_explicit(self):
        rows=[{"measurement":{"milestones_s":{str(n):n*3 if n<=18 else None for n in range(1,31)}}},
              {"measurement":{"milestones_s":{str(n):n*4 if n<=20 else None for n in range(1,31)}}},
              {"error":"failed trial"}]
        summary={r["milestone"]:r for r in aggregate_milestones(rows)}
        self.assertEqual((summary[18]["reached"],summary[18]["median_s"]),(2,63))
        self.assertEqual((summary[20]["reached"],summary[20]["median_s"]),(1,80))
        self.assertIsNone(summary[22]["median_s"])
        self.assertEqual(summary[22]["declared_trials"],3)

    def test_invalid_identity_is_retained_without_scored_milestones(self):
        row = {"identity_valid": False, "measurement": {"milestones_s": {str(n): n*3 for n in range(1,31)}}}
        result = aggregate_milestones([row])
        self.assertTrue(all(r["reached"] == 0 for r in result))
        self.assertTrue(all(r["identity_invalid_trials"] == 1 for r in result))

    def test_vote_ties_abstain_and_missing_verdicts_are_not_scored(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp)/"trace/01";folder.mkdir(parents=True)
            row={"problem_idx":1,"rollouts":[{"rollout":i,"status":"completed","finish_reason":"stop","extracted_candidate":"70" if i<=2 else "69"} for i in range(1,5)]}
            atomic_json(folder/"question.json",row)
            (folder/"verification.jsonl").write_text('\n'.join(json.dumps({"candidate":candidate,"result":{"verdict":verdict}}) for candidate,verdict in (("70",True),("69",False)))+'\n')
            accuracy=baseline_accuracy(tmp,4)
            self.assertTrue(accuracy["valid"])
            self.assertEqual(accuracy["majority_vote_correct_questions"],0)
            self.assertTrue(accuracy["per_question"][0]["vote_tied"])
            (folder/"verification.jsonl").write_text('')
            accuracy=baseline_accuracy(tmp,4)
            self.assertFalse(accuracy["valid"])
            self.assertIsNone(accuracy["pass1"])


if __name__=="__main__":unittest.main()
