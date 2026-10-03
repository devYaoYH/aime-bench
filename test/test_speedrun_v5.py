"""Offline verdict-aware allocation and exact observed-prefix resumption."""
import asyncio
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
from src.attempt_runners.speedrun_v5 import parse_args, run_speedrun
from test.test_attempt import Stream
from test.test_speedrun_v1 import capped


def partial(prompt, tokens, text, *, completion=False, ids=True):
    choice = {'index': 0, 'finish_reason': None}
    if ids:
        choice['token_ids'] = tokens
    choice.update({'text': text} if completion else {'delta': {'content': text}})
    return [('data: '+json.dumps({'prompt_token_ids': prompt, 'choices': [choice],
                                'usage': {'completion_tokens': len(tokens)}})+'\n\n').encode()]


class VerdictAwareTests(unittest.IsolatedAsyncioTestCase):
    async def exercise(self, handler, *, count=1, target=1, changes=None):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            args = parse_args(['--model','r0b0tlab/VibeThinker-3B-NVFP4','--benchmark',
                               '--target-correct',str(target),'--question-timeout','2'])
            for k,v in (changes or {}).items():
                setattr(args,k,v)
            store=AttemptArtifacts(output,buffered=True)
            profiler=AttemptProfiler(output,enabled=False,artifacts=store)
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                rows=await asyncio.wait_for(run_speedrun(
                    [{'problem_idx':q,'problem':str(q)} for q in range(1,count+1)],
                    args,client,output,DisabledGPUSampler(),time.perf_counter(),profiler),3)
            allocation=store.read_json(output/'allocation.json')
            assert not (output/'allocation.json').exists()
            data={}
            store.flush()
            for p in output.rglob('*.json'):
                data[str(p.relative_to(output))]=json.loads(p.read_text())
            return rows,allocation,data

    async def test_close_stream_before_verify_and_resume_exact_ids_after_wrong(self):
        requests=[];streams=[];checks=[]
        async def handler(request):
            body=json.loads(request.content)
            if request.url.path=='/verify':
                self.assertTrue(all(s.closed for s in streams))
                checks.append(body['candidate'])
                await asyncio.sleep(.003)
                return httpx.Response(200,json={'verdict':body['candidate']=='71'})
            requests.append(body)
            if len(requests)==1:
                chunks=partial([101,102],[400,401],'\\boxed{070}')
            else:
                self.assertEqual(body['prompt'],[101,102,400,401])
                self.assertEqual(body['max_tokens'],6)
                chunks=partial(body['prompt'],[402],'\\boxed{071}',completion=True)
            stream=Stream(chunks,hang=True);streams.append(stream)
            return httpx.Response(200,stream=stream)
        rows,a,data=await self.exercise(handler,changes={'max_concurrent_requests':1,'token_budgets':[8,16],
                                                       'first_pass_max_tokens':8,'max_tokens':16})
        self.assertEqual(checks,['70','71']);self.assertEqual(len(requests),2)
        self.assertEqual(rows[0]['status'],'solved')
        self.assertTrue(all(r['status']=='paused' for r in rows[0]['rollouts']))
        self.assertEqual([r['continuation_of_rollout'] for r in rows[0]['rollouts']],[None,1])
        self.assertTrue(data['trace/01/rollout-01/tokens.json']['observed_prefix_complete'])
        self.assertEqual(a['questions']['1']['used'],2)

    async def test_pending_question_releases_slot_without_new_samples_for_it(self):
        requests=[];streams=[];expanded=asyncio.Event();active_checks=set()
        async def handler(request):
            body=json.loads(request.content)
            if request.url.path=='/verify':
                q=body['index'];self.assertNotIn(q,active_checks);active_checks.add(q)
                self.assertTrue(streams[q-1].closed)
                if q==1:
                    await asyncio.wait_for(expanded.wait(),.5)
                    self.assertEqual(sum(x[0]==1 for x in requests),1)
                await asyncio.sleep(.002);active_checks.remove(q)
                return httpx.Response(200,json={'verdict':True})
            q=int(body['messages'][1]['content']);r=body['seed']-20261003-q*4
            requests.append((q,r))
            chunks=partial([101],[400],'\\boxed{070}' if q==1 or r>1 else 'thinking')
            stream=Stream(chunks,delay=.001,hang=True);streams.append(stream)
            if q==2 and r>1:expanded.set()
            return httpx.Response(200,stream=stream)
        rows,a,_=await self.exercise(handler,count=2,target=2,changes={'max_concurrent_requests':2})
        self.assertTrue(expanded.is_set());self.assertEqual(sum(r['status']=='solved' for r in rows),2)
        self.assertLessEqual(a['peak_active_requests'],2)
        self.assertTrue(all(s.closed for s in streams))

    async def test_two_candidates_in_one_chunk_checked_serially_without_resume(self):
        streams=[];checks=[];requests=[]
        async def handler(request):
            body=json.loads(request.content)
            if request.url.path=='/verify':
                self.assertTrue(streams[0].closed);checks.append(body['candidate'])
                return httpx.Response(200,json={'verdict':body['candidate']=='71'})
            requests.append(body);stream=Stream(partial([101],[400,401],'\\boxed{070} \\boxed{071}'),hang=True)
            streams.append(stream);return httpx.Response(200,stream=stream)
        rows,a,_=await self.exercise(handler,changes={'max_concurrent_requests':1})
        self.assertEqual(checks,['70','71']);self.assertEqual(len(requests),1)
        self.assertEqual(rows[0]['status'],'solved')

    async def test_four_request_cap_includes_repeated_paused_resumptions(self):
        requests=[];streams=[]
        async def handler(request):
            body=json.loads(request.content)
            if request.url.path=='/verify':
                return httpx.Response(200,json={'verdict':False})
            requests.append(body);prompt=body.get('prompt',[101]);n=len(requests)
            stream=Stream(partial(prompt,[400+n],f'\\boxed{{{n}}}',completion='prompt' in body),hang=True)
            streams.append(stream);return httpx.Response(200,stream=stream)
        rows,a,_=await self.exercise(handler,changes={'max_concurrent_requests':1})
        self.assertEqual(len(requests),4);self.assertEqual(a['questions']['1']['used'],4)
        self.assertEqual(rows[0]['status'],'unsolved');self.assertTrue(all(s.closed for s in streams))
        self.assertEqual([len(r.get('prompt',[])) for r in requests],[0,2,3,4])

    async def test_missing_ids_fail_after_wrong_but_not_after_correct(self):
        async def scenario(correct):
            streams=[]
            async def handler(request):
                if request.url.path=='/verify':return httpx.Response(200,json={'verdict':correct})
                stream=Stream(partial([101],[],'\\boxed{070}',ids=False),hang=True);streams.append(stream)
                return httpx.Response(200,stream=stream)
            try:
                return await self.exercise(handler,changes={'max_concurrent_requests':1})
            finally:
                self.assertTrue(all(s.closed for s in streams))
        rows,_,_=await scenario(True);self.assertEqual(rows[0]['status'],'solved')
        with self.assertRaisesRegex(RuntimeError,'exact observed token IDs'):
            await scenario(False)

    async def test_capped_output_preserves_cumulative_budget_ladder(self):
        requests=[]
        async def handler(request):
            self.assertNotEqual(request.url.path,'/verify')
            body=json.loads(request.content);requests.append(body);prompt=body.get('prompt',[101])
            tokens=[400+i for i in range(body['max_tokens'])]
            return httpx.Response(200,stream=Stream(capped(prompt,tokens,'thinking',completion='prompt' in body)))
        rows,a,_=await self.exercise(handler,changes={'token_budgets':[2,4,8,16],
                                'first_pass_max_tokens':2,'max_tokens':16,'max_concurrent_requests':1})
        self.assertEqual([r['max_tokens'] for r in requests],[2,2,4,8])
        self.assertEqual(rows[0]['status'],'unsolved');self.assertEqual(a['questions']['1']['used'],4)

    async def test_target_reached_closes_unfinished_streams(self):
        streams=[]
        async def handler(request):
            body=json.loads(request.content)
            if request.url.path=='/verify':return httpx.Response(200,json={'verdict':True})
            q=int(body['messages'][1]['content'])
            stream=Stream(partial([101],[400],'\\boxed{070}' if q==1 else 'thinking'),hang=True)
            streams.append(stream);return httpx.Response(200,stream=stream)
        rows,a,_=await self.exercise(handler,count=3,changes={'max_concurrent_requests':3})
        self.assertEqual(sum(r['status']=='solved' for r in rows),1)
        self.assertTrue(all(s.closed for s in streams))
        self.assertTrue(all(v['used']<=4 for v in a['questions'].values()))

    async def test_full_30_initial_questions_target18_and_cap30(self):
        requests=[];streams=[];oracle=asyncio.Lock()
        async def handler(request):
            body=json.loads(request.content)
            if request.url.path=='/verify':
                async with oracle:
                    await asyncio.sleep(.001)
                    return httpx.Response(200,json={'verdict':True})
            q=int(body['messages'][1]['content']);r=body['seed']-20261003-q*4
            requests.append((q,r))
            stream=Stream(partial([101],[400],'\\boxed{070}' if q<=6 or r>1 else 'thinking'),hang=True)
            streams.append(stream);return httpx.Response(200,stream=stream)
        rows,a,_=await self.exercise(handler,count=30,target=18)
        self.assertEqual(len(rows),30)
        self.assertEqual(requests[:30],[(q,1) for q in range(1,31)])
        self.assertEqual(a['peak_active_requests'],30)
        self.assertTrue(all(r['active_requests']<=30 for r in a['admissions']))
        self.assertTrue(all(v['used']<=4 for v in a['questions'].values()))
        self.assertEqual(sum(r['status']=='solved' for r in rows),18)
        self.assertTrue(all(s.closed for s in streams))

    async def test_generation_failure_closes_other_streams(self):
        streams=[]
        async def handler(request):
            body=json.loads(request.content)
            if body['messages'][1]['content']=='1':
                return httpx.Response(500)
            stream=Stream(partial([101],[400],'thinking'),hang=True);streams.append(stream)
            return httpx.Response(200,stream=stream)
        with self.assertRaisesRegex(RuntimeError,'500'):
            await self.exercise(handler,count=3,target=2,changes={'max_concurrent_requests':3})
        self.assertTrue(all(s.closed for s in streams))

    async def test_target_cancels_another_question_pending_verdict(self):
        streams=[];other_pending=asyncio.Event()
        async def handler(request):
            body=json.loads(request.content)
            if request.url.path=='/verify':
                if body['index']==1:
                    await other_pending.wait()
                    return httpx.Response(200,json={'verdict':True})
                other_pending.set()
                await asyncio.Event().wait()
            stream=Stream(partial([101],[400],'\\boxed{070}'),hang=True);streams.append(stream)
            return httpx.Response(200,stream=stream)
        rows,a,_=await self.exercise(handler,count=2,changes={'max_concurrent_requests':2})
        self.assertEqual([r['status'] for r in rows],['solved','stopped'])
        self.assertTrue(all(s.closed for s in streams))
        self.assertTrue(all(s['closed'] and not s['pending'] for s in a['questions'].values()))

    def test_policy_defaults_and_cap_are_isolated(self):
        args=parse_args(['--model','r0b0tlab/VibeThinker-3B-NVFP4','--benchmark'])
        self.assertEqual((args.max_concurrent_requests,args.max_attempts_per_question),(30,4))
        self.assertEqual(args.token_budgets,[8192,16384])
        self.assertEqual(args.schedule,'suspend_pending_verdict')
        self.assertEqual((args.benchmark_year,args.benchmark_role),(2025,'development'))
        with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):
            parse_args(['--model','r0b0tlab/VibeThinker-3B-NVFP4','--max-attempts-per-question','5'])
