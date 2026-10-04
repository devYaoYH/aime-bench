"""Frozen-source integrity and configuration-independent solving invariants."""
import ast
import asyncio
import json
from pathlib import Path
import tempfile
import time
import unittest

import httpx

from runner_final import run as historical, _streaming as old_stream
from runner_final.core_v1 import runner, _streaming
from runner_final.integrity import MANIFEST, verify_core
from runner_final.run_frozen import parse_args
from src.common import ROOT
from src.attempt_storage import AttemptArtifacts, DisabledGPUSampler
from src.attempt_metrics import AttemptProfiler
from test.test_attempt import FakeGPU, Stream, chunk
from test.test_runner_final import capped


class FrozenBoundaryTests(unittest.TestCase):
    def test_verified_manifest_and_source_drift_rejected(self):
        self.assertEqual(len(verify_core(ROOT)),64)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/MANIFEST;manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({'schema_version':1,'core_id':'runner_final_core_v1','sha256':{'engine.py':'0'*64}}))
            (root/'engine.py').write_text('changed source')
            with self.assertRaisesRegex(RuntimeError,'Frozen core drift.*engine.py'):
                verify_core(root)

    def test_configuration_changes_leave_core_hash_and_policy_unchanged(self):
        digest=verify_core(ROOT)
        baseline=parse_args([])
        stronger=parse_args(['--preset',str(ROOT/'runner_final/presets/prompt_adherence.json')])
        shorter=parse_args(['--preset',str(ROOT/'runner_final/presets/30x2_4k.json')])
        self.assertEqual(baseline.system_prompt,old_stream.PROMPT)
        self.assertNotEqual(baseline.system_prompt_sha256,stronger.system_prompt_sha256)
        self.assertEqual((shorter.parallelism,shorter.rollouts,shorter.first_pass_max_tokens,shorter.max_attempts_per_question),(30,2,4096,4))
        self.assertEqual(verify_core(ROOT),digest)
        self.assertTrue(baseline.skip_benchmark_prewarm)
        self.assertTrue(baseline.benchmark)
        self.assertEqual(parse_args(['--seed','17']).seed,17)
        with tempfile.TemporaryDirectory() as tmp:
            prompt=Path(tmp)/'prompt.txt';prompt.write_text('')
            with self.assertRaisesRegex(ValueError,'cannot be empty'):
                parse_args(['--system-prompt-file',str(prompt)])

    def test_frozen_scheduling_and_parser_match_historical_core(self):
        def definition(module,name):
            tree=ast.parse(Path(module.__file__).read_text())
            return ast.dump(next(n for n in tree.body if getattr(n,'name',None)==name),include_attributes=False)
        for name in ('group_options','question_state','run_speedrun'):
            self.assertEqual(definition(runner,name),definition(historical,name))
        for name in ('CandidateDetector','continuation_prefix'):
            self.assertEqual(definition(_streaming,name),definition(old_stream,name))


class FrozenStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def test_prompt_is_input_and_wrong_box_keeps_generation_running(self):
        checks,requests,streams=[],[],[]
        async def handler(request):
            body=json.loads(request.content)
            if request.url.path=='/v1/chat/completions':
                requests.append(body)
                stream=Stream([chunk('\\boxed{98}\n'),chunk('recounting'),chunk('\\boxed{82}\n')],delay=.003,hang=True)
                streams.append(stream);return httpx.Response(200,stream=stream)
            checks.append(body['candidate']);await asyncio.sleep(.015)
            return httpx.Response(200,json={'verdict':body['candidate']=='82'})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);args=parse_args(['--target-correct','1']);args.system_prompt='Emit a candidate immediately.'
            profiler=AttemptProfiler(root,enabled=False,artifacts=AttemptArtifacts(root,buffered=True))
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                result=await _streaming.run_question({'problem_idx':18,'problem':'Count colorings.'},args,client,root,FakeGPU(),profiler=profiler)
            self.assertEqual(requests[0]['messages'][0]['content'],args.system_prompt)
            self.assertEqual(checks,['98','82']);self.assertEqual(result['winner']['answer'],82)
            self.assertTrue(streams[0].closed)

    async def test_split_answer_survives_exact_id_continuation_in_buffered_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);store=AttemptArtifacts(root,buffered=True);requests=[]
            profiler=AttemptProfiler(root,enabled=False,artifacts=store)
            args=parse_args(['--target-correct','1'])
            async def handler(request):
                body=json.loads(request.content)
                self.assertEqual(list(root.rglob('*')),[])
                if request.url.path=='/verify':
                    self.assertEqual(body['candidate'],'70');return httpx.Response(200,json={'verdict':True})
                requests.append(body)
                if len(requests)==1:return httpx.Response(200,stream=Stream(capped([10],[11],'\\boxed{0')))
                self.assertEqual(body['prompt'],[10,11])
                return httpx.Response(200,stream=Stream(capped([10,11],[12],'70}',completion=True)))
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                rows=await runner.run_speedrun([{'problem_idx':1,'problem':'test'}],args,client,root,DisabledGPUSampler(),time.perf_counter(),profiler)
            self.assertEqual(rows[0]['status'],'solved');self.assertEqual(len(requests),2)
            self.assertEqual(list(root.rglob('*')),[])
            store.flush()
            self.assertEqual(len((root/'solved.jsonl').read_text().splitlines()),1)


if __name__=='__main__':unittest.main()
