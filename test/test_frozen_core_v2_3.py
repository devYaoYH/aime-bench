"""Input-sized slot reuse, long context budgets, and feedback scheduling races."""
import asyncio
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import AsyncMock, patch

import httpx

from runner_final.core_v2_3 import runner, _streaming, budgets
from runner_final.core_v2_3.allocation import AllocationPool
from runner_final.run_frozen_v2_3 import parse_args
from src.attempt_storage import AttemptArtifacts, DisabledGPUSampler
from src.attempt_metrics import AttemptProfiler
from src.common import ROOT
from test.test_attempt import Stream
from test.test_frozen_core_v2_2 import Validator


def chunk(text, prompt=(1,2,3), ids=(10,), finish=None, completion=False, reasoning=None):
    choice={'index':0,'delta':{'content':text}, 'token_ids':list(ids),'finish_reason':finish}
    if completion: choice['text']=text
    if reasoning is not None:choice['delta']['reasoning_content']=reasoning
    body={'prompt_token_ids':list(prompt),'usage':{'completion_tokens':len(ids)},'choices':[choice]}
    return ('data: '+json.dumps(body)+'\n\n').encode()


class PoolRunner(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.out=Path(self.tmp.name)
        self.requests=[];self.checks=[];self.streams=[];self.tokenized=[]
        self.args=parse_args(['--seed','20261011','--question-timeout','2'])
        self.artifacts=AttemptArtifacts(self.out,buffered=True)
        self.profiler=AttemptProfiler(self.out,enabled=False,artifacts=self.artifacts)

    async def asyncTearDown(self):self.tmp.cleanup()

    def stream(self, chunks, *, hang=True):
        s=Stream(chunks,delay=.001,hang=hang);self.streams.append(s)
        return httpx.Response(200,stream=s)

    def qid(self,body):
        return (body['seed']-self.args.seed-1)//4  # seed index*4+request; request=1..4

    async def run_case(self, handler, *, count=3, target=2, problems=None, validator=None):
        self.args.target_correct=target
        problems=problems if problems is not None else [{'problem_idx':q,'problem':str(q),'prompt_tokens':3} for q in range(1,count+1)]
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with redirect_stdout(io.StringIO()):
                rows=await asyncio.wait_for(runner.run_speedrun(problems,self.args,client,
                    self.out,DisabledGPUSampler(),time.perf_counter(),self.profiler,
                    syntax_validator=validator or Validator()),3)
        allocation=self.artifacts.read_json(self.out/'allocation.json')
        self.artifacts.flush()
        self.assertTrue(all(len(r['rollouts'])<=4 for r in rows))
        return rows,allocation

    async def handler(self,request):
        body=json.loads(request.content)
        if request.url.path=='/verify':
            self.checks.append(body)
            return httpx.Response(200,json={'verdict':body['candidate']=='3'})
        if request.url.path=='/tokenize':
            self.tokenized.append(body)
            if 'messages' in body:return httpx.Response(200,json={'count':6,'tokens':list(range(6))})
            return httpx.Response(200,json={'count':2,'tokens':[90,91]})
        self.requests.append((request.url.path,body))
        answer=r'\boxed{1}' if 'messages' in body else r'\boxed{3}'
        return self.stream([chunk(answer,prompt=body.get('prompt',[1,2,3]),completion='prompt' in body)])

    async def test_all_input_questions_start_once_at_dynamic_slot_count(self):
        async def handler(request):
            body=json.loads(request.content)
            if request.url.path=='/v1/chat/completions':
                self.requests.append((request.url.path,body))
                return self.stream([chunk(r'\boxed{3}')])
            return await self.handler(request)
        rows,allocation=await self.run_case(handler,count=47,target=47)
        self.assertEqual((self.args.parallelism,self.args.max_concurrent_requests),(47,47))
        self.assertEqual(allocation['peak_active_requests'],47)
        self.assertEqual([self.qid(b) for _,b in self.requests[:47]],list(range(1,48)))
        self.assertTrue(all(b['max_tokens']==65533 for _,b in self.requests[:47]))
        self.assertEqual(sum(r['status']=='solved' for r in rows),47)
        self.assertTrue(all(s.closed for s in self.streams))

    async def test_solved_slots_recycle_to_siblings_without_barrier(self):
        oracle=asyncio.Lock()
        async def handler(request):
            body=json.loads(request.content)
            if request.url.path=='/verify':
                async with oracle:
                    await asyncio.sleep(.002)
                    self.checks.append(body)
                    return httpx.Response(200,json={'verdict':True})
            self.requests.append((request.url.path,body))
            q=self.qid(body);request_no=body['seed']-self.args.seed-q*4
            text=r'\boxed{3}' if q==1 or request_no>=2 else 'thinking'
            return self.stream([chunk(text)])
        rows,allocation=await self.run_case(handler,count=3,target=3)
        self.assertEqual(allocation['peak_active_requests'],3)
        self.assertTrue(any(a['rollout']>1 for a in allocation['admissions']))
        self.assertTrue(any(s['peak_active']>1 for s in allocation['questions'].values()))
        self.assertTrue(all(a['active_requests']<=3 for a in allocation['admissions']))
        self.assertTrue(all(n<=4 for n in Counter(self.qid(b) for _,b in self.requests).values()))
        self.assertEqual(sum(r['status']=='solved' for r in rows),3)
        self.assertTrue(all(s.closed for s in self.streams))

    async def test_natural_end_fresh_retries_all_have_long_budget(self):
        async def handler(request):
            body=json.loads(request.content);self.requests.append((request.url.path,body))
            return self.stream([chunk('thinking',finish='stop'),b'data: [DONE]\n\n'],hang=False)
        rows,allocation=await self.run_case(handler,count=3,target=2)
        self.assertEqual(len(self.requests),12)
        self.assertTrue(all(u=='/v1/chat/completions' and b['max_tokens']==65533 for u,b in self.requests))
        self.assertTrue(all(r['status']=='unsolved' for r in rows))
        self.assertTrue(all(s['used']==4 for s in allocation['questions'].values()))
        self.assertIsNone(allocation['first_submission'])

    async def test_context_cap_causes_fresh_retry_not_short_segments(self):
        self.args.max_context_tokens=8
        async def handler(request):
            body=json.loads(request.content);self.requests.append((request.url.path,body))
            return self.stream([chunk('thinking',ids=range(body['max_tokens']),finish='length'),b'data: [DONE]\n\n'],hang=False)
        rows,allocation=await self.run_case(handler,count=1,target=1)
        self.assertEqual([b['max_tokens'] for _,b in self.requests],[5]*4)
        self.assertTrue(all(u=='/v1/chat/completions' for u,_ in self.requests))
        self.assertEqual(rows[0]['status'],'unsolved')
        self.assertEqual(allocation['questions']['1']['used'],4)

    async def test_feedback_fork_is_pool_admission_with_remaining_long_context(self):
        rows,allocation=await self.run_case(self.handler,count=1,target=1)
        self.assertEqual(rows[0]['status'],'solved')
        self.assertEqual(len(self.requests),2)
        self.assertEqual(self.requests[1][1]['prompt'],[1,2,3,10,90,91])
        self.assertEqual(self.requests[1][1]['max_tokens'],65536-6)
        self.assertEqual([a['kind'] for a in allocation['admissions']],['fresh','correction'])
        self.assertEqual(allocation['peak_active_requests'],1)
        self.assertEqual(rows[0]['rollouts'][1]['lane_id'],1)
        self.assertTrue(all(s.closed for s in self.streams))

    async def test_feedback_for_multiple_questions_shares_global_slot_limit(self):
        rows,allocation=await self.run_case(self.handler,count=3,target=3)
        self.assertEqual(sum(r['status']=='solved' for r in rows),3)
        self.assertTrue(any(a['kind']=='correction' for a in allocation['admissions']))
        self.assertTrue(all(a['active_requests']<=3 for a in allocation['admissions']))
        self.assertEqual(allocation['peak_active_requests'],3)

    async def test_repeated_feedback_exhausts_four_requests_without_extra_fork(self):
        async def handler(request):
            body=json.loads(request.content)
            if request.url.path=='/verify':
                self.checks.append(body)
                return httpx.Response(200,json={'verdict':False})
            if request.url.path=='/tokenize':return await self.handler(request)
            self.requests.append((request.url.path,body))
            number=len(self.requests)
            return self.stream([chunk(r'\boxed{'+str(number)+'}',prompt=body.get('prompt',[1,2,3]),
                               completion='prompt' in body,finish='stop' if number==4 else None)],hang=number!=4)
        rows,allocation=await self.run_case(handler,count=1,target=1)
        self.assertEqual([r['candidate'] for r in self.checks],['1','2','3','4'])
        self.assertEqual(rows[0]['status'],'unsolved')
        self.assertEqual(allocation['questions']['1']['used'],4)
        self.assertEqual(len(self.requests),4)
        self.assertEqual([b['max_tokens'] for _,b in self.requests],[65533,65530,65527,65524])

    async def test_grader_failure_is_not_negative_feedback(self):
        async def handler(request):
            if request.url.path=='/verify':return httpx.Response(500)
            return await self.handler(request)
        with self.assertRaisesRegex(RuntimeError,'failed'):
            await self.run_case(handler,count=1,target=1)
        self.assertEqual(self.tokenized,[])
        self.assertTrue(all(s.closed for s in self.streams))

    async def test_actual_cpu_validation_and_expression_dedup_preserved(self):
        from runner_final.core_v2_3.syntax import ExpressionValidator
        validator=ExpressionValidator()
        await validator.start()
        async def handler(request):
            if request.url.path=='/v1/chat/completions':
                self.requests.append((request.url.path,json.loads(request.content)))
                return self.stream([chunk(r'\boxed{EXPRESSION}\boxed{4/8}\boxed{\frac{1}{2}}\boxed{3}')])
            return await self.handler(request)
        try:
            rows,_=await self.run_case(handler,count=1,target=1,validator=validator)
        finally:validator.close()
        self.assertEqual(rows[0]['status'],'solved')
        self.assertEqual([r['candidate'] for r in self.checks],['4/8','3'])
        audit=[json.loads(line) for line in (self.out/'trace/01/candidate_validation.jsonl').read_text().splitlines()]
        self.assertEqual([r['outcome'] for r in audit],['rejected','enqueued','duplicate_expression','enqueued'])
        self.assertEqual(self.tokenized,[])

    async def test_queued_self_correction_avoids_fork(self):
        async def handler(request):
            if request.url.path=='/v1/chat/completions':
                self.requests.append((request.url.path,json.loads(request.content)))
                return self.stream([chunk(r'\boxed{1}\boxed{3}')])
            return await self.handler(request)
        rows,allocation=await self.run_case(handler,count=1,target=1)
        self.assertEqual(rows[0]['status'],'solved');self.assertEqual(len(self.requests),1)
        self.assertEqual(self.tokenized,[])
        self.assertEqual([r['candidate'] for r in self.checks],['1','3'])

    async def test_multiple_wrong_candidates_form_one_batch(self):
        async def handler(request):
            if request.url.path=='/v1/chat/completions':
                self.requests.append((request.url.path,json.loads(request.content)))
                return self.stream([chunk(r'\boxed{1}\boxed{2}')])
            return await self.handler(request)
        rows,allocation=await self.run_case(handler,count=1,target=1)
        self.assertEqual(rows[0]['status'],'solved')
        self.assertEqual([r['candidate'] for r in self.checks],['1','2','3'])
        self.assertEqual(len(self.tokenized),1);self.assertIn('["1", "2"]',self.tokenized[0]['prompt'])
        self.assertEqual(allocation['questions']['1']['used'],2)

    async def test_validation_pending_in_both_channels_prevents_preemption(self):
        graded=asyncio.Event()
        async def validate(answer):
            if answer=='3':await graded.wait()
        async def handler(request):
            if request.url.path=='/v1/chat/completions':
                self.requests.append((request.url.path,json.loads(request.content)))
                return self.stream([chunk(r'\boxed{3}',reasoning=r'\boxed{1}')])
            response=await self.handler(request)
            if request.url.path=='/verify':graded.set()
            return response
        rows,_=await self.run_case(handler,count=1,target=1,validator=Validator(validate))
        self.assertEqual(rows[0]['status'],'solved');self.assertEqual(self.tokenized,[])
        self.assertEqual(len(self.requests),1)

    async def test_candidate_during_feedback_tokenization_defers_fork(self):
        preparing,arrived=asyncio.Event(),asyncio.Event()
        class Later(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield chunk(r'\boxed{1}')
                await preparing.wait();yield chunk(r'\boxed{3}')
                await asyncio.Event().wait()
            async def aclose(self):pass
        async def validate(answer):
            if answer=='3':arrived.set()
        async def handler(request):
            if request.url.path=='/v1/chat/completions':
                self.requests.append((request.url.path,json.loads(request.content)))
                return httpx.Response(200,stream=Later())
            if request.url.path=='/tokenize':
                preparing.set();await arrived.wait();await asyncio.sleep(0)
            return await self.handler(request)
        rows,allocation=await self.run_case(handler,count=1,target=1,validator=Validator(validate))
        self.assertEqual(rows[0]['status'],'solved');self.assertEqual(len(self.requests),1)
        self.assertEqual(allocation['questions']['1']['used'],1)

    async def test_wrong_after_finished_stream_is_feedback_in_fresh_prompt(self):
        owner=self
        class EndAfterVerdict(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield chunk('thinking',finish='stop')
                path=owner.out/'trace/01/feedback.jsonl'
                while not any(r['kind']=='wrong_verdict' for r in owner.artifacts.jsonl_files.get(path,('',[]))[1]):
                    await asyncio.sleep(0)
                yield chunk('',ids=(),finish='stop')
                yield b'data: [DONE]\n\n'
            async def aclose(self):pass
        async def handler(request):
            body=json.loads(request.content)
            if request.url.path=='/v1/chat/completions':
                self.requests.append((request.url.path,body))
                if len(self.requests)==2:
                    return httpx.Response(200,stream=EndAfterVerdict())
                return self.stream([chunk(r'\boxed{1}' if len(self.requests)==1 else r'\boxed{3}',finish='stop'),b'data: [DONE]\n\n'],hang=False)
            if request.url.path=='/verify':
                await asyncio.sleep(.01)  # allocator can recycle while grader is busy
            return await self.handler(request)
        rows,_=await self.run_case(handler,count=1,target=1)
        self.assertEqual(rows[0]['status'],'solved')
        fresh_with_feedback=[b for u,b in self.requests if u=='/v1/chat/completions'
                             and 'External grader feedback' in b['messages'][1]['content']]
        self.assertTrue(fresh_with_feedback)
        self.assertTrue(all(b['max_tokens']==65536-6 for b in fresh_with_feedback))

    async def test_stream_failure_aborts_and_closes_peers(self):
        async def handler(request):
            body=json.loads(request.content)
            if body['messages'][1]['content']=='1':return httpx.Response(500)
            return self.stream([chunk('thinking')])
        began=time.perf_counter()
        with self.assertRaisesRegex(RuntimeError,'500'):
            await self.run_case(handler,count=3,target=2)
        self.assertLess(time.perf_counter()-began,.5)
        self.assertTrue(all(s.closed for s in self.streams))

    async def test_missing_ids_cannot_continue_capped_trajectory(self):
        async def handler(request):
            return self.stream([b'data: {"choices":[{"delta":{"content":"thinking"},"finish_reason":"length"}]}\n\n',b'data: [DONE]\n\n'],hang=False)
        with self.assertRaisesRegex(RuntimeError,'token-ID evidence'):
            await self.run_case(handler,count=1,target=1)

    async def test_global_cancellation_during_preparation_closes_every_stream(self):
        preparing=asyncio.Event()
        async def handler(request):
            if request.url.path=='/tokenize':preparing.set();await asyncio.Event().wait()
            return await self.handler(request)
        task=asyncio.create_task(self.run_case(handler,count=1,target=1))
        await preparing.wait();task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertTrue(all(s.closed for s in self.streams))

    async def test_empty_or_duplicate_question_indices_rejected(self):
        for problems in ([],[{'problem_idx':1,'problem':'1'}]*2):
            with self.assertRaisesRegex(ValueError,'unique question'):
                await self.run_case(self.handler,problems=problems)


class AllocationPolicy(unittest.IsolatedAsyncioTestCase):
    async def test_same_ready_then_least_active_rotating_priority_as_v15(self):
        from src.attempt_runners._allocation_v1_5 import AllocationPool as Original
        args=parse_args([])
        for cls in (Original,AllocationPool):
            pool=cls([1,2,3],args,None,None,0)
            pool.states[1]['active']=2;pool.states[2]['active']=1
            pool.states[1]['ready'].append({'continuation':{}})
            self.assertEqual(pool.choose(),1)
            pool.states[1]['ready'].clear()
            self.assertEqual(pool.choose(),3)
            pool.states[3]['active']=1
            self.assertEqual(pool.choose(),2)

    async def test_paused_question_does_not_lose_last_slot_to_fresh_admission(self):
        args=parse_args([]);pool=AllocationPool([1,2],args,None,None,0)
        pool.states[1]['used']=3;pool.pause(1,True)
        self.assertEqual(pool.choose(),2)
        pool.queue_correction(1,{'continuation':{'prompt':[1]},'stage':1,'target_generated_tokens':65536})
        self.assertFalse(pool.queue_correction(1,{}))
        pool.pause(1,False);self.assertEqual(pool.choose(),1)


class BudgetAndContract(unittest.IsolatedAsyncioTestCase):
    async def test_all_freezes_and_prompt_parser_contracts_preserved(self):
        import hashlib
        from runner_final import integrity, integrity_v2, integrity_v2_1, integrity_v2_2, integrity_v2_3
        from runner_final.core_v2_1 import syntax as original
        from runner_final.core_v2_3 import syntax, extraction
        self.assertEqual(syntax.POLICY,original.POLICY)
        self.assertEqual((ROOT/'runner_final/core_v2_1/extraction.py').read_bytes(),Path(extraction.__file__).read_bytes())
        self.assertEqual(parse_args([]).system_prompt_sha256,
                         hashlib.sha256((ROOT/'runner_final/prompts/math_core_v2.txt').read_bytes()).hexdigest())
        for verifier in (integrity,integrity_v2,integrity_v2_1,integrity_v2_2,integrity_v2_3):
            self.assertEqual(len(verifier.verify_core(ROOT)),64)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);manifest=root/integrity_v2_3.MANIFEST
            manifest.parent.mkdir(parents=True)
            manifest.write_text(json.dumps({'schema_version':1,'core_id':runner.RUNNER_ID,'sha256':{'policy.py':'0'*64}}))
            (root/'policy.py').write_text('changed')
            with self.assertRaisesRegex(RuntimeError,'Frozen core drift.*policy.py'):
                integrity_v2_3.verify_core(root)

    async def test_served_template_count_and_thinking_configuration(self):
        args=parse_args(['--disable-thinking']);requests=[]
        async def handler(request):
            body=json.loads(request.content);requests.append(body)
            return httpx.Response(200,json={'count':3,'tokens':[11,12,13]})
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            problems=await budgets.prepare_prompts([{'problem_idx':91,'problem':'Compute.'}],args,client)
        self.assertEqual(problems[0]['prompt_tokens'],3)
        self.assertFalse(requests[0]['add_special_tokens'])
        self.assertTrue(requests[0]['add_generation_prompt'])
        self.assertEqual(requests[0]['chat_template_kwargs'],{'enable_thinking':False})
        self.assertEqual(requests[0]['messages'][0]['content'],args.system_prompt)

    async def test_invalid_prompt_counts_fail_before_generation(self):
        args=parse_args([])
        for value in ({'count':0,'tokens':[]},{'count':3,'tokens':[1]},
                      {'count':1,'tokens':[True]},{'count':65536,'tokens':[]}):
            async def handler(request):return httpx.Response(200,json=value)
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                with self.assertRaisesRegex(ValueError,'prompt token evidence'):
                    await budgets.prepare_prompts([{'problem_idx':1,'problem':'Compute.'}],args,client)

    def test_cli_defines_input_sized_long_request_policy(self):
        args=parse_args([])
        self.assertEqual((args.max_tokens,args.first_pass_max_tokens,args.max_context_tokens),(65536,)*3)
        self.assertIsNone(args.parallelism);self.assertIsNone(args.max_concurrent_requests)
        self.assertEqual((args.rollouts,args.max_attempts_per_question,args.schedule),(1,4,'eager_pool'))
        for argv in (['--parallelism','30'],['--rollouts','2'],['--max-tokens','16384'],
                     ['--first-pass-max-tokens','8192'],['--schedule','barrier'],['--max-attempts-per-question','5']):
            with redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):parse_args(argv)
