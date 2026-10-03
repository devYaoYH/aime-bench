"""Offline staged fan-out ordering, cancellation, dedup and request-cap checks."""
import asyncio
import json
from pathlib import Path
import tempfile
import time
import unittest
import httpx
from src.attempt_metrics import AttemptProfiler
from src.attempt_storage import AttemptArtifacts, DisabledGPUSampler
from src.attempt_runners.speedrun_v3 import parse_args, run_speedrun
from test.test_attempt import Stream, chunk
from test.test_speedrun_v1 import capped


class StagedTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, handler, count=30, target=18):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            args=parse_args(['--model','WeiboAI/VibeThinker-3B','--benchmark',
                             '--target-correct',str(target),'--question-timeout','2'])
            store=AttemptArtifacts(out,buffered=True)
            profiler=AttemptProfiler(out,enabled=False,artifacts=store)
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                rows=await asyncio.wait_for(run_speedrun(
                    [{'problem_idx':q,'problem':str(q)} for q in range(1,count+1)],
                    args,client,out,DisabledGPUSampler(),time.perf_counter(),profiler),3)
            fanout=store.read_json(out/'fanout.json')
            self.assertFalse((out/'fanout.json').exists())
            store.flush()
            self.assertEqual(len(rows),count)
            self.assertTrue(all(len(r['rollouts'])<=4 for r in rows))
            return rows,fanout

    async def test_all_initial_before_trigger_all_four_before_verdict_and_dedup(self):
        requests,checks,streams=[],[],[]
        full=asyncio.Event();oracle=asyncio.Lock()
        async def handler(request):
            body=json.loads(request.content)
            if request.url.path=='/verify':
                if not checks:self.assertEqual(len(requests),30)
                await full.wait()
                async with oracle:
                    checks.append(body)
                    await asyncio.sleep(.002)
                    return httpx.Response(200,json={'verdict':True})
            requests.append(body)
            q=int(body['messages'][1]['content'])
            stream=Stream([chunk('\\boxed{070}' if q<=18 else 'thinking')],delay=.001,hang=True)
            streams.append(stream)
            if len(requests)==120:full.set()
            return httpx.Response(200,stream=stream)
        rows,f=await self.exercise(handler)
        self.assertEqual(len(requests),120);self.assertEqual(len(checks),18)
        self.assertEqual(f['first_submission']['initial_requests_started'],30)
        self.assertEqual(len(f['question_expansions']),30)
        self.assertTrue(all(x['additional_requests']==3 for x in f['question_expansions'].values()))
        self.assertTrue(all(len(r['rollouts'])==4 for r in rows))
        self.assertTrue(all(s.closed for s in streams))
        self.assertEqual(sum(r['status']=='solved' for r in rows),18)
        for q in range(1,31):
            same=[x for x in requests if x['messages'][1]['content']==str(q)]
            self.assertEqual({x['seed'] for x in same},{20261003+q*4+r for r in range(1,5)})
            self.assertEqual({x['max_tokens'] for x in same},{8192})

    async def test_no_candidates_exhausts_initial_pass_without_hanging(self):
        requests=[]
        async def handler(request):
            self.assertNotEqual(request.url.path,'/verify')
            body=json.loads(request.content);requests.append(body)
            return httpx.Response(200,stream=Stream(capped([1],[50],'thinking')))
        rows,f=await self.exercise(handler,count=3,target=2)
        self.assertEqual(len(requests),3);self.assertIsNone(f['first_submission'])
        self.assertFalse(f['question_expansions'])
        self.assertTrue(all(r['status']=='unsolved' for r in rows))

    async def test_wrong_first_verdict_still_expands_and_serializes_per_question(self):
        requests=[];active=set();checks=[]
        async def handler(request):
            body=json.loads(request.content)
            if request.url.path=='/verify':
                q=body['index'];self.assertNotIn(q,active);active.add(q)
                try:
                    await asyncio.sleep(.005);checks.append(body)
                    return httpx.Response(200,json={'verdict':body['candidate']=='70'})
                finally:active.remove(q)
            requests.append(body);q=int(body['messages'][1]['content'])
            first=body['seed']==20261003+q*4+1
            return httpx.Response(200,stream=Stream(capped([q],[51],'\\boxed{111}' if first else '\\boxed{070}'),delay=.001))
        rows,f=await self.exercise(handler,count=3,target=3)
        self.assertEqual(len(requests),12);self.assertEqual(len(checks),6)
        self.assertEqual(sum(r['status']=='solved' for r in rows),3)
        self.assertEqual(f['first_submission']['candidate'],'111')

    async def test_stream_failure_cancels_other_questions(self):
        streams=[]
        async def handler(request):
            body=json.loads(request.content)
            if body['messages'][1]['content']=='1':return httpx.Response(500)
            s=Stream([chunk('thinking')],hang=True);streams.append(s)
            return httpx.Response(200,stream=s)
        began=time.perf_counter()
        with self.assertRaisesRegex(RuntimeError,'Q1 failed'):
            await self.exercise(handler,count=3,target=2)
        self.assertLess(time.perf_counter()-began,.5)
        self.assertTrue(all(s.closed for s in streams))

    def test_cli_rejects_partial_fanout_or_larger_cap(self):
        for argv in (['--rollouts','2'],['--max-attempts-per-question','5']):
            with self.assertRaises(SystemExit):parse_args(['--model','WeiboAI/VibeThinker-3B',*argv])
