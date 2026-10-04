"""Policy and deferred sweep checks; no GPU, SSH, or model access."""

import asyncio
from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import time
import socket
import unittest
from unittest.mock import patch

import httpx

from runner_final.run import parse_args, run_speedrun
from src.common import ROOT
from test.test_attempt import FakeGPU, Stream, chunk
from runner_final import run as final_runner, prewarm, validate
from src.attempt_runners import speedrun_v2
from src.attempt_metrics import AttemptProfiler
from src.attempt_storage import AttemptArtifacts, DisabledGPUSampler
from unittest.mock import AsyncMock


class FinalSettingsTests(unittest.TestCase):
    def test_defaults_match_best_policy_except_identity_and_prewarm(self):
        from src.experiments.replicate_best_v1 import controls
        expected = vars(speedrun_v2.parse_args(controls('r0b0tlab/VibeThinker-3B-NVFP4', model_profile='vllm-flashinfer.yaml')))
        actual = vars(parse_args(['--benchmark', '--reuse-server']))
        actual.pop('skip_benchmark_prewarm')
        self.assertEqual(actual.pop('prewarm_max_tokens'), 8192)
        actual['strategy'] = expected['strategy']
        self.assertEqual(actual, expected)
        protocol = json.loads(validate.PROTOCOL.read_text())
        self.assertEqual(len(protocol['seeds']), 5)
        self.assertEqual(len(set(protocol['seeds'])), 5)
        self.assertTrue(all(s > 20261007 for s in protocol['seeds']))

    def test_refuses_fifth_request_and_warming_test_year(self):
        for argv in (['--max-attempts-per-question','5'], ['--benchmark-year','2024'], ['--seed','-1'], ['--prewarm-max-tokens','0']):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(argv)
        self.assertEqual(parse_args(['--benchmark-year','2024','--skip-benchmark-prewarm']).benchmark_year, 2024)

    def test_confirmation_includes_failed_first_trial(self):
        rows = [{'valid':True,'time_to_target_s':60}] * 4
        self.assertFalse(validate.aggregate(rows,71)['confirmed'])
        rows = [{'valid':False,'time_to_target_s':None}] + rows
        result = validate.aggregate(rows,71)
        self.assertFalse(result['confirmed'])
        self.assertEqual(result['successful_trials'],4)
        self.assertTrue(validate.aggregate([{'valid':True,'time_to_target_s':60}] * 5,71)['confirmed'])


class PrewarmTests(unittest.IsolatedAsyncioTestCase):
    async def test_five_seeds_driver_retains_failure_and_runs_later_seeds(self):
        from types import SimpleNamespace
        seen=[];closed=[]
        class Services:
            def launch(self,*a,**kw):return object()
            async def close(self):closed.append(True)
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root=Path(tmp);(root/'attempts').mkdir()
            profile=root/'models/r0b0tlab/VibeThinker-3B-NVFP4/vllm-flashinfer.yaml'
            profile.parent.mkdir(parents=True);profile.write_text(Path(final_runner.__file__).with_name('vllm-flashinfer.yaml').read_text())
            original_parse=final_runner.parse_args
            def parse(argv):return original_parse(argv+['--models-dir',str(root/'models')])
            async def run(args):
                seen.append(args.seed)
                output=root/'attempts'/str(args.seed);output.mkdir()
                if len(seen)==3:raise RuntimeError('trial failed')
                return output
            def score(output,seed,reference):
                return {'attempt_id':output.name,'sampling_seed':seed,'valid':True,'time_to_target_s':60}
            with patch.object(validate,'ROOT',root),patch.object(validate,'Services',Services),patch.object(validate,'ensure_free'),patch.object(final_runner,'assert_gpu_idle'),patch.object(final_runner,'parse_args',new=parse),patch.object(final_runner,'run',new=run),patch.object(validate,'score',new=score),patch.object(validate,'ready',new=AsyncMock()),patch.object(validate,'reset_cache',new=AsyncMock(return_value={'success':True})),patch.object(validate.subprocess,'check_output',side_effect=['','f'*40]):
                batch=await validate.execute(SimpleNamespace(batch='five-seed-test'))
            result=json.loads((batch/'summary.json').read_text())
            self.assertEqual(seen,json.loads(validate.PROTOCOL.read_text())['seeds'])
            self.assertEqual(len(result['trials']),5)
            self.assertEqual(result['trials'][2]['status'],'failed')
            self.assertEqual(result['trials'][2]['attempt_id'],str(seen[2]))
            self.assertEqual(result['successful_trials'],4)
            self.assertFalse(result['confirmed'])
            self.assertEqual(closed,[True])

    async def test_all30_2024_statements_no_grader_and_clear_prefixes(self):
        requests, resets, active, peak = [], [], 0, 0
        async def handler(request):
            nonlocal active, peak
            if request.url.path == '/reset_prefix_cache':
                self.assertEqual(json.loads(request.content or b'{}'), {})
                resets.append(request.url.path)
                return httpx.Response(200,json={'success':True})
            self.assertEqual(request.url.path,'/v1/chat/completions')
            body=json.loads(request.content);requests.append(body)
            active+=1;peak=max(peak,active)
            await asyncio.sleep(.005)
            active-=1
            return httpx.Response(200,json={'choices':[{'message':{'content':'irrelevant wrong answer'},'finish_reason':'length'}], 'usage':{'completion_tokens':8192}})
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result=await prewarm.prewarm_benchmark(parse_args([]),client,Path(tmp))
            self.assertEqual((len(requests),peak,len(resets)),(30,30,2))
            self.assertEqual(result['grader_queries'],0)
            self.assertFalse(result['correctness_evaluated'])
            from src.benchmarks import load_questions
            self.assertEqual([r['messages'][1]['content'] for r in requests],[q['problem'] for q in load_questions(year=2024)])
            self.assertEqual({r['max_tokens'] for r in requests},{8192})
            self.assertEqual(len(list(Path(tmp).glob('prewarming-2024/trace/*/response.json'))),30)
            self.assertTrue((Path(tmp)/'prewarm.json').exists())
            self.assertFalse(list(Path(tmp).glob('trace/*')))

    async def test_failed_prewarm_cancels_outstanding_requests(self):
        cancelled=[];started=asyncio.Event();count=0
        async def handler(request):
            nonlocal count
            if request.url.path=='/reset_prefix_cache':
                return httpx.Response(200,json={'success':True})
            count+=1;slot=count
            if count==30:started.set()
            await started.wait()
            if slot==1:return httpx.Response(500)
            try:await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancelled.append(slot);raise
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                with self.assertRaises(httpx.HTTPStatusError):
                    await asyncio.wait_for(prewarm.prewarm_benchmark(parse_args([]),client,Path(tmp)),2)
            self.assertEqual(len(cancelled),29)
            self.assertFalse((Path(tmp)/'prewarm.json').exists())

    async def test_managed_prewarm_precedes_grader_and_official_timing(self):
        launches=[];order=[]
        class Process:
            returncode=0
            def wait(self):return 0
        class Services:
            def launch(self,command,*a,**kw):launches.append(command);return Process()
            async def close(self):order.append('cleanup')
        async def warm(args,client,output):
            self.assertFalse(any('server.py' in ' '.join(cmd) for cmd in launches))
            self.assertFalse((output/'config.json').read_text().find('official_started_at_utc')>=0)
            order.append('prewarm')
            return {'total_latency_s':3, 'grader_queries':0}
        async def solve(problems,args,client,output,gpu,started,profiler):
            self.assertTrue(any('server.py' in ' '.join(cmd) for cmd in launches))
            self.assertEqual(order,['prewarm'])
            order.append('solve')
            profiler.artifacts.write_json(output/'trace/01/question.json',{'problem_idx':1,'status':'stopped','end_to_end_latency_s':0,'unique_candidates':0,'winner':None,'round':1,'rollouts':[],'first_solved':None})
            return profiler.artifacts.questions()
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
            root=Path(tmp);profile=root/'models/r0b0tlab/VibeThinker-3B-NVFP4/vllm-flashinfer.yaml';profile.parent.mkdir(parents=True)
            import yaml
            profile.write_text(yaml.safe_dump({'max-model-len':65536,'override-generation-config':'{"max_new_tokens":16384}'}))
            args=parse_args(['--benchmark','--target-correct','1','--models-dir',str(root/'models')])
            health={'queries_so_far':0,'cost_c':3,'dataset':{'sha256':final_runner.dataset_provenance(2025)['grader_sha256']}}
            with patch.object(final_runner,'ROOT',root),patch.object(final_runner,'Services',Services),patch.object(final_runner,'ensure_free'),patch.object(final_runner,'assert_gpu_idle'),patch.object(final_runner.subprocess,'check_output',return_value='f'*40),patch.object(final_runner,'ready',new=AsyncMock(side_effect=[{'data':[{'id':args.model,'max_model_len':65536}]},health])),patch.object(final_runner,'prewarm_benchmark',new=warm),patch.object(final_runner,'warm_inference',new=AsyncMock(return_value={'latency_s':0,'batch_size':30,'tokens_per_request':32})),patch.object(final_runner,'load_questions',return_value=[{'problem_idx':1,'problem':'test'}]),patch.object(final_runner,'run_speedrun',new=solve):
                output=await final_runner.run(args)
            self.assertEqual(order,['prewarm','solve','cleanup'])
            result=json.loads((output/'summary.json').read_text())
            self.assertEqual(result['benchmark_prewarm']['grader_queries'],0)
            self.assertFalse(result['target_reached'])

    async def test_buffered_continuation_exact_ids_before_flush(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);store=AttemptArtifacts(output,buffered=True)
            profiler=AttemptProfiler(output,enabled=False,artifacts=store)
            args=parse_args(['--benchmark','--target-correct','1']);requests=[]
            async def handler(request):
                self.assertEqual(list(output.rglob('*')),[])
                body=json.loads(request.content)
                if request.url.path=='/verify':
                    self.assertEqual(body['candidate'],'70');return httpx.Response(200,json={'verdict':True})
                requests.append(body)
                if len(requests)==1:return httpx.Response(200,stream=Stream(capped([10],[11],'\\boxed{0')))
                self.assertEqual(body['prompt'],[10,11])
                return httpx.Response(200,stream=Stream(capped([10,11],[12],'70}',completion=True)))
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                rows=await run_speedrun([{'problem_idx':1,'problem':'test'}],args,client,output,DisabledGPUSampler(),time.perf_counter(),profiler)
            self.assertEqual(rows[0]['status'],'solved')
            self.assertEqual(len(requests),2)
            self.assertEqual(list(output.rglob('*')),[])
            store.flush()
            self.assertEqual(len((output/'solved.jsonl').read_text().splitlines()),1)


def capped(prompt, tokens, text, *, completion=False):
    choice = {"index": 0, "token_ids": tokens, "finish_reason": "length"}
    choice.update(
        {"text": text, "prompt_token_ids": prompt}
        if completion
        else {"delta": {"content": text}}
    )
    row = {
        "choices": [choice],
        "usage": {"prompt_tokens": len(prompt), "completion_tokens": len(tokens)},
    }
    if not completion:
        row["prompt_token_ids"] = prompt
    return [("data: " + json.dumps(row) + "\n\n").encode(), b"data: [DONE]\n\n"]


class FrozenFinalPolicyTests(unittest.IsolatedAsyncioTestCase):
    def args(self, **changes):
        args = parse_args(
            ["--model", "WeiboAI/VibeThinker-3B", "--question-timeout", "2"]
        )
        args.rollouts = 4  # Exercise preserved fan-out controls beyond final default one.
        for k, v in changes.items():
            setattr(args, k, v)
        return args

    async def test_all_120_initial_requests_target18_cancellation_and_budget4(self):
        streams, requests, checks = [], [], []
        all_started = asyncio.Event()
        oracle = asyncio.Lock()

        async def handler(request):
            body = json.loads(request.content)
            if request.url.path == "/verify":
                await all_started.wait()
                async with oracle:
                    checks.append(body)
                    await asyncio.sleep(0.001)
                    return httpx.Response(200, json={"verdict": True})
            requests.append(body)
            q = int(body["messages"][1]["content"])
            stream = Stream(
                [chunk("\\boxed{070}" if q <= 18 else "thinking")],
                delay=0.001,
                hang=True,
            )
            streams.append(stream)
            if len(streams) == 120:
                all_started.set()
            return httpx.Response(200, stream=stream)

        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                rows = await asyncio.wait_for(
                    run_speedrun(
                        [{"problem_idx": q, "problem": str(q)} for q in range(1, 31)],
                        self.args(),
                        client,
                        Path(tmp),
                        FakeGPU(),
                        time.perf_counter(),
                    ),
                    3,
                )
            self.assertEqual(len(requests), 120)
            self.assertEqual(
                len(checks), 18
            )  # four identical candidates => one verification/question
            self.assertEqual(len(rows), 30)
            self.assertEqual(sum(r["status"] == "solved" for r in rows), 18)
            self.assertEqual(sum(r["status"] == "stopped" for r in rows), 12)
            self.assertTrue(all(len(r["rollouts"]) == 4 for r in rows))
            self.assertTrue(all(s.closed for s in streams))
            self.assertEqual({r["max_tokens"] for r in requests}, {8192})
            self.assertEqual(
                len((Path(tmp) / "solved.jsonl").read_text().splitlines()), 18
            )

    async def test_pass2_continues_distinct_lanes_and_cap4(self):
        requests = []

        async def handler(request):
            self.assertNotEqual(request.url.path, "/verify")
            body = json.loads(request.content)
            requests.append((request.url.path, body))
            if request.url.path == "/v1/chat/completions":
                prompt, tokens = [body["seed"]], [50]
            else:
                prompt, tokens = body["prompt"], [51]
            return httpx.Response(
                200,
                stream=Stream(
                    capped(prompt, tokens, "thinking", completion="prompt" in body)
                ),
            )

        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                rows = await run_speedrun(
                    [{"problem_idx": 1, "problem": "test"}],
                    self.args(rollouts=2, target_correct=1),
                    client,
                    Path(tmp),
                    FakeGPU(),
                    time.perf_counter(),
                )
            self.assertEqual(len(requests), 4)
            self.assertEqual(
                [r[0] for r in requests],
                ["/v1/chat/completions"] * 2 + ["/v1/completions"] * 2,
            )
            self.assertEqual(requests[2][1]["prompt"], [requests[0][1]["seed"], 50])
            self.assertEqual(requests[3][1]["prompt"], [requests[1][1]["seed"], 50])
            self.assertNotEqual(requests[2][1]["prompt"], requests[3][1]["prompt"])
            self.assertEqual(
                [r[1]["max_tokens"] for r in requests], [8192, 8192, 16384, 16384]
            )
            self.assertEqual(
                [r["continuation_of_rollout"] for r in rows[0]["rollouts"]],
                [None, None, 1, 2],
            )
            self.assertEqual(rows[0]["status"], "unsolved")

    async def test_fifo_fresh_before_retries_and_eager_vs_barrier(self):
        async def schedule(kind):
            order = []
            release = asyncio.Event()
            q2_started = asyncio.Event()

            async def handler(request):
                body = json.loads(request.content)
                q = int(body["messages"][1]["content"])
                order.append(q)
                if q == 2 and order.count(2) == 1:
                    q2_started.set()
                    await release.wait()
                return httpx.Response(
                    200,
                    stream=Stream(
                        [chunk("none", finish="stop"), b"data: [DONE]\n\n"], delay=0.001
                    ),
                )

            with tempfile.TemporaryDirectory() as tmp:
                async with httpx.AsyncClient(
                    transport=httpx.MockTransport(handler)
                ) as client:
                    task = asyncio.create_task(
                        run_speedrun(
                            [{"problem_idx": q, "problem": str(q)} for q in (1, 2, 3)],
                            self.args(
                                schedule=kind,
                                rollouts=1,
                                parallelism=2,
                                target_correct=3,
                                max_rounds=2,
                                no_continuation=True,
                            ),
                            client,
                            Path(tmp),
                            FakeGPU(),
                            time.perf_counter(),
                        )
                    )
                    await q2_started.wait()
                    for _ in range(50):
                        if order.count(1) >= 2 or (kind == "barrier" and 3 in order):
                            break
                        await asyncio.sleep(0.001)
                    self.assertEqual(order[:3], [1, 2, 3])
                    self.assertEqual(order.count(1) >= 2, kind == "eager")
                    release.set()
                    await asyncio.wait_for(task, 2)
            self.assertEqual(len(order), 6)

        await schedule("eager")
        await schedule("barrier")

    async def test_worker_failure_does_not_hang_or_retry(self):
        requests = []

        async def handler(request):
            requests.append(request)
            return httpx.Response(500)

        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                with self.assertRaises(RuntimeError):
                    await asyncio.wait_for(
                        run_speedrun(
                            [{"problem_idx": 1, "problem": "test"}],
                            self.args(rollouts=1),
                            client,
                            Path(tmp),
                            FakeGPU(),
                            time.perf_counter(),
                        ),
                        1,
                    )
            row = json.loads((Path(tmp) / "trace/01/question.json").read_text())
            self.assertEqual(row["status"], "error")
            self.assertEqual(len(requests), 1)

    async def test_pass1_counts_continuations_as_requests_and_stitches_candidate(self):
        requests = []

        async def handler(request):
            body = json.loads(request.content)
            if request.url.path == "/verify":
                self.assertEqual(body["candidate"], "70")
                return httpx.Response(200, json={"verdict": True})
            requests.append(body)
            if len(requests) == 1:
                return httpx.Response(
                    200, stream=Stream(capped([10], [11], "\\boxed{0"))
                )
            return httpx.Response(
                200, stream=Stream(capped([10, 11], [12], "70}", completion=True))
            )

        with tempfile.TemporaryDirectory() as tmp:
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(handler)
            ) as client:
                rows = await run_speedrun(
                    [{"problem_idx": 1, "problem": "test"}],
                    self.args(rollouts=1, target_correct=1),
                    client,
                    Path(tmp),
                    FakeGPU(),
                    time.perf_counter(),
                )
            self.assertEqual(len(requests), 2)
            self.assertEqual(rows[0]["status"], "solved")
            self.assertEqual(requests[1]["prompt"], [10, 11])
