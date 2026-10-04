"""Offline dynamic allocation, growing cumulative budgets and cleanup checks."""
import asyncio
from collections import Counter
from contextlib import redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
import httpx
from src.attempt_metrics import AttemptProfiler
from src.attempt_storage import AttemptArtifacts, DisabledGPUSampler
from src.attempt_runners.speedrun_v1_5 import run_speedrun
from runner_final.run_v1_5 import parse_args
from test.test_attempt import Stream, chunk
from test.test_speedrun_v1 import capped


class Eager30Tests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, handler, *, count=30, target=18, changes=None):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            args=parse_args(['--model','r0b0tlab/VibeThinker-3B-NVFP4','--benchmark',
                             '--target-correct',str(target),'--question-timeout','2'])
            for k,v in (changes or {}).items():setattr(args,k,v)
            store=AttemptArtifacts(out,buffered=True)
            profiler=AttemptProfiler(out,enabled=False,artifacts=store)
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                rows=await asyncio.wait_for(run_speedrun(
                    [{'problem_idx':q,'problem':str(q)} for q in range(1,count+1)],
                    args,client,out,DisabledGPUSampler(),time.perf_counter(),profiler),3)
            allocation=store.read_json(out/'allocation.json')
            self.assertFalse((out/'allocation.json').exists())
            store.flush()
            self.assertTrue(all(len(r['rollouts'])<=args.max_attempts_per_question for r in rows))
            return rows,allocation


    async def test_cap30_recycles_solved_slots_with_original_four_request_cap(self):
        requests, streams, checks = [], [], []
        oracle = asyncio.Lock()
        async def handler(request):
            body = json.loads(request.content)
            if request.url.path == '/verify':
                async with oracle:
                    await asyncio.sleep(.002)
                    checks.append(body)
                    return httpx.Response(200, json={'verdict': True})
            q = int(body['messages'][1]['content'])
            r = body['seed'] - 20261003 - q * 4
            requests.append((q, r))
            stream = Stream([chunk('\\boxed{070}' if q <= 6 or r >= 2 else 'thinking')], delay=.001, hang=True)
            streams.append(stream)
            return httpx.Response(200, stream=stream)
        rows, allocation = await self.exercise(handler, changes={
            'max_concurrent_requests': 30, 'max_attempts_per_question': 4,
            'token_budgets': [8192, 16384], 'max_tokens': 16384})
        self.assertEqual(allocation['peak_active_requests'], 30)
        self.assertEqual(len(checks),18)
        self.assertEqual(len({b['index'] for b in checks}),18)
        self.assertTrue(all(r == 1 for _, r in requests[:30]))
        self.assertTrue(any(r > 1 for _, r in requests[30:]))
        self.assertTrue(all(a['active_requests'] <= 30 for a in allocation['admissions']))
        self.assertTrue(all(n <= 4 for n in Counter(q for q, _ in requests).values()))
        self.assertEqual(sum(r['status'] == 'solved' for r in rows), 18)
        self.assertTrue(all(s.closed for s in streams))


    async def test_stream_failure_aborts_fast_and_closes_other_streams(self):
        streams=[]
        async def handler(request):
            body=json.loads(request.content)
            if body['messages'][1]['content']=='1':return httpx.Response(500)
            s=Stream([chunk('thinking')],hang=True);streams.append(s)
            return httpx.Response(200,stream=s)
        began=time.perf_counter()
        with self.assertRaisesRegex(RuntimeError,'500'):
            await self.exercise(handler,count=3,target=2)
        self.assertLess(time.perf_counter()-began,.5)
        self.assertTrue(all(s.closed for s in streams))


    async def test_missing_exact_ids_refuses_continuation(self):
        async def handler(request):
            return httpx.Response(200,stream=Stream([chunk('thinking',finish='length')]))
        with self.assertRaisesRegex(RuntimeError,'token-ID evidence'):
            await self.exercise(handler,count=1,target=1)


    async def test_v1_additional_budget_and_exact_prefix(self):
        requests=[]
        async def handler(request):
            body=json.loads(request.content); requests.append((request.url.path,body))
            prompt=body.get('prompt',[101,102,103])
            tokens=list(range(500+len(requests)*20,500+len(requests)*20+body['max_tokens']))
            return httpx.Response(200,stream=Stream(capped(prompt,tokens,'thinking',completion='prompt' in body)))
        rows,a=await self.exercise(handler,count=1,target=1,changes={
            'first_pass_max_tokens':2,'max_tokens':4,'max_context_tokens':16,
            'max_concurrent_requests':1})
        self.assertEqual([b['max_tokens'] for _,b in requests],[2,4,4,3])
        self.assertEqual([u for u,_ in requests],['/v1/chat/completions']+['/v1/completions']*3)
        self.assertEqual([len(b.get('prompt',[])) for _,b in requests],[0,5,9,13])
        self.assertEqual(requests[2][1]['prompt'][:5],requests[1][1]['prompt'])
        self.assertEqual(rows[0]['status'],'unsolved')
        self.assertEqual(a['questions']['1']['used'],4)
        self.assertEqual(requests[0][1]['messages'][0]['content'],parse_args([]).system_prompt)

    async def test_natural_completion_retries_before_any_grader_submission(self):
        requests=[]
        async def handler(request):
            body=json.loads(request.content);requests.append(body)
            return httpx.Response(200,stream=Stream([chunk('thinking',finish='stop')]))
        rows,a=await self.exercise(handler,count=3,target=2,changes={'max_concurrent_requests':3})
        self.assertEqual(len(requests),12)
        self.assertTrue(all('messages' in b for b in requests))
        self.assertEqual([b['max_tokens'] for b in requests[:3]],[8192]*3)
        self.assertTrue(all(b['max_tokens']==16384 for b in requests[3:]))
        self.assertTrue(all(r['status']=='unsolved' and len(r['rollouts'])==4 for r in rows))
        self.assertIsNone(a['first_submission'])

    def test_unchanged_control_settings_and_manifests(self):
        from runner_final.run_frozen import parse_args as original
        from runner_final.integrity import verify_core
        from runner_final.integrity_v1_5 import verify_core as verify_new
        from src.common import ROOT
        old=original(['--preset','runner_final/presets/prompt_adherence.json'])
        new=parse_args([])
        for key in ('system_prompt','temperature','top_p','seed','first_pass_max_tokens',
                    'max_tokens','max_attempts_per_question','model','model_profile',
                    'benchmark','skip_benchmark_prewarm','grader_cost'):
            self.assertEqual(getattr(old,key),getattr(new,key),key)
        self.assertEqual(new.max_concurrent_requests,30)
        verify_core(ROOT);verify_new(ROOT)
