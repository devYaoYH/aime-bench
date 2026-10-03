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
from src.attempt_runners.speedrun_v4 import parse_args, run_speedrun
from test.test_attempt import Stream, chunk
from test.test_speedrun_v1 import capped


class DynamicTests(unittest.IsolatedAsyncioTestCase):
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

    async def test_exact_id_continuations_cumulative_budgets_and_context_clip(self):
        requests=[]
        async def handler(request):
            self.assertNotEqual(request.url.path,'/verify')
            body=json.loads(request.content);requests.append((request.url.path,body))
            prompt=body.get('prompt',[101,102,103]);tokens=list(range(500+len(requests)*20,500+len(requests)*20+body['max_tokens']))
            return httpx.Response(200,stream=Stream(capped(prompt,tokens,'thinking',completion='prompt' in body)))
        rows,a=await self.exercise(handler,count=1,target=1,changes={
            'token_budgets':[2,4,8,16],'first_pass_max_tokens':2,'max_tokens':16,
            'max_context_tokens':16,'max_concurrent_requests':1,'max_attempts_per_question':4})
        self.assertEqual([b['max_tokens'] for _,b in requests],[2,2,4,5])
        self.assertEqual([u for u,_ in requests],['/v1/chat/completions']+['/v1/completions']*3)
        self.assertEqual([len(b.get('prompt',[])) for _,b in requests],[0,5,7,11])
        self.assertEqual(requests[2][1]['prompt'][:5],requests[1][1]['prompt'])
        self.assertEqual(rows[0]['status'],'unsolved');self.assertEqual(a['peak_active_requests'],1)
        self.assertEqual([x['target_generated_tokens'] for x in a['admissions']],[2,4,8,16])

    async def test_cap60_first30_then_fresh_reallocation_and_question_dedup(self):
        requests,streams,checks=[],[],[]
        full=asyncio.Event();oracle=asyncio.Lock();active_verify=set()
        async def handler(request):
            body=json.loads(request.content)
            if request.url.path=='/verify':
                q=body['index'];self.assertNotIn(q,active_verify);active_verify.add(q)
                try:
                    if not full.is_set():self.assertEqual(len(requests),30)
                    await full.wait()
                    async with oracle:
                        await asyncio.sleep(.002);checks.append(body)
                        return httpx.Response(200,json={'verdict':True})
                finally:active_verify.remove(q)
            self.assertIn('messages',body)
            q=int(body['messages'][1]['content']);r=body['seed']-20261003-q*4
            requests.append((q,r,body))
            answer=q<=6 or r>=3
            stream=Stream([chunk('\\boxed{070}' if answer else 'thinking')],delay=.001,hang=True)
            streams.append(stream)
            if len(requests)==60:full.set()
            return httpx.Response(200,stream=stream)
        rows,a=await self.exercise(handler)
        self.assertEqual(a['peak_active_requests'],60)
        self.assertEqual(len(checks),18);self.assertEqual(sum(r['status']=='solved' for r in rows),18)
        self.assertTrue(all(r==1 for _,r,_ in requests[:30]))
        self.assertTrue(any(s['peak_active']>2 for s in a['questions'].values()))
        self.assertTrue(all(x['active_requests']<=60 for x in a['admissions']))
        self.assertTrue(all(s.closed for s in streams))
        self.assertTrue(all(b['max_tokens']==8192 for _,_,b in requests))
        self.assertTrue(all(n<=8 for n in Counter(q for q,_,_ in requests).values()))

    async def test_no_candidates_exhausts_eight_requests_without_deadlock(self):
        requests=[]
        async def handler(request):
            self.assertNotEqual(request.url.path,'/verify')
            body=json.loads(request.content);requests.append(body)
            prompt=body.get('prompt',[101]);tokens=[400+i for i in range(body['max_tokens'])]
            return httpx.Response(200,stream=Stream(capped(prompt,tokens,'thinking',completion='prompt' in body)))
        rows,a=await self.exercise(handler,count=3,target=2,changes={
            'token_budgets':[1,2],'first_pass_max_tokens':1,'max_tokens':2,'max_context_tokens':100})
        self.assertEqual(len(requests),24);self.assertIsNone(a['first_submission'])
        self.assertEqual(a['peak_active_requests'],3)
        self.assertTrue(all(len(r['rollouts'])==8 and r['status']=='unsolved' for r in rows))
        self.assertEqual(sum('prompt' in r for r in requests),12)

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
        with self.assertRaisesRegex(RuntimeError,'complete exact token IDs'):
            await self.exercise(handler,count=1,target=1)

    def test_invalid_budget_and_concurrency_controls(self):
        for extra in (['--max-attempts-per-question','9'],['--rollouts','2'],
                      ['--token-budgets','8192','4096'],['--max-concurrent-requests','0']):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(['--model','test/model',*extra])
