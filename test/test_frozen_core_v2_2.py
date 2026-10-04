"""Queue-aware runtime corrections against timed fake streaming services."""
import asyncio
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import httpx

from runner_final.core_v2_2 import runner, _streaming
from runner_final.run_frozen_v2_2 import parse_args
from runner_final.core_v2_2.feedback import feedback_text
from src.attempt_storage import AttemptArtifacts
from test.test_attempt import FakeGPU, Stream


def tokens_chunk(text, *, ids=True, count=1, finish=None, reasoning=None):
    choice = {'index':0, 'delta': {'content':text}, 'finish_reason':finish}
    if reasoning is not None: choice['delta']['reasoning_content'] = reasoning
    body = {'choices':[choice]}
    choice['text'] = text
    if ids:
        body['prompt_token_ids'] = [1,2]
        choice['token_ids'] = [10] * count
        body['usage'] = {'completion_tokens':count}
    return ('data: '+json.dumps(body)+'\n\n').encode()


class Validator:
    def __init__(self, callback=None): self.callback=callback
    async def validate(self, answer):
        if self.callback: await self.callback(answer)
        return {'valid':True,'canonical_key':answer,'cache_hit':True,
                'validation_wall_s':0,'queue_s':0,'syntax_cpu_s':0,'canonical_cpu_s':0}


class Corrections(unittest.IsolatedAsyncioTestCase):
    async def run_case(self, handler, *, changes=None, validator=None, output=None, round_no=1, offset=0, prefixes=None):
        args=parse_args(['--question-timeout','2'])
        args.continuation_max_tokens=args.max_tokens
        args.max_tokens=args.first_pass_max_tokens if round_no==1 else args.max_tokens
        for key,value in (changes or {}).items(): setattr(args,key,value)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with redirect_stdout(io.StringIO()):
                return await _streaming.run_question({'problem_idx':1,'problem':'Compute.'}, args,
                    client, output or self.output, FakeGPU(), syntax_validator=validator or Validator(),
                    round_no=round_no, rollout_offset=offset, continuations=prefixes)

    async def asyncSetUp(self):
        self.tmp=tempfile.TemporaryDirectory(); self.output=Path(self.tmp.name)
        self.requests=[]; self.streams=[]; self.checked=[]; self.feedback=[]

    async def asyncTearDown(self): self.tmp.cleanup()

    def response_stream(self, values, hang=True):
        stream=Stream(values,delay=0,hang=hang); self.streams.append(stream)
        return httpx.Response(200,stream=stream)

    async def simple_handler(self, request):
        body=json.loads(request.content)
        if request.url.path=='/verify':
            self.checked.append(body['candidate'])
            return httpx.Response(200,json={'verdict':body['candidate']=='3','query_id':'query'})
        if request.url.path=='/tokenize':
            self.feedback.append(body);return httpx.Response(200,json={'tokens':[90,91],'count':2})
        self.requests.append(body)
        return self.response_stream([tokens_chunk(r'\boxed{1}' if 'messages' in body else r'\boxed{3}')])

    async def test_wrong_forks_exact_prefix_with_16k_budget_and_distinct_seed(self):
        result=await self.run_case(self.simple_handler)
        self.assertEqual(result['status'],'solved',result)
        self.assertEqual(self.checked,['1','3'])
        self.assertEqual(len(self.requests),2)
        self.assertEqual(self.requests[1]['prompt'],[1,2,10,90,91])
        self.assertEqual(self.requests[1]['max_tokens'],16384)
        self.assertNotEqual(self.requests[0]['seed'],self.requests[1]['seed'])
        self.assertEqual(self.requests[0]['max_tokens'],8192)
        self.assertFalse(self.feedback[0]['add_special_tokens'])
        self.assertEqual(result['rollouts'][1]['continuation_of_rollout'],1)
        self.assertEqual(result['rollouts'][1]['lane_id'],1)
        self.assertTrue(all(s.closed for s in self.streams))
        self.assertEqual(result['rejected_answers'],['1'])
        events=[json.loads(l) for l in (self.output/'trace/01/feedback.jsonl').read_text().splitlines()]
        self.assertEqual([e['kind'] for e in events].count('forked'),1)

    async def test_queued_correct_prevents_fork_even_when_same_chunk(self):
        async def handler(request):
            if request.url.path=='/v1/chat/completions':
                body=json.loads(request.content);self.requests.append(body)
                return self.response_stream([tokens_chunk(r'\boxed{1}\boxed{3}')])
            return await self.simple_handler(request)
        result=await self.run_case(handler)
        self.assertEqual(result['status'],'solved',result)
        self.assertEqual(self.checked,['1','3'])
        self.assertEqual(len(self.requests),1);self.assertEqual(self.feedback,[])

    async def test_multiple_queued_wrong_answers_are_batched(self):
        async def handler(request):
            if request.url.path=='/v1/chat/completions':
                self.requests.append(json.loads(request.content))
                return self.response_stream([tokens_chunk(r'\boxed{1}\boxed{2}')])
            return await self.simple_handler(request)
        await self.run_case(handler)
        self.assertEqual(self.checked,['1','2','3'])
        self.assertEqual(len(self.feedback),1)
        self.assertIn('["1", "2"]',self.feedback[0]['prompt'])

    async def test_validation_in_flight_cannot_be_preempted(self):
        first_graded=asyncio.Event()
        async def validate(answer):
            if answer=='3': await first_graded.wait()
        async def handler(request):
            if request.url.path=='/v1/chat/completions':
                self.requests.append(json.loads(request.content))
                return self.response_stream([tokens_chunk(r'\boxed{1}\boxed{3}')])
            response=await self.simple_handler(request)
            if request.url.path=='/verify': first_graded.set()
            return response
        result=await self.run_case(handler,validator=Validator(validate))
        self.assertEqual(result['status'],'solved',result);self.assertEqual(self.feedback,[])
        self.assertEqual(len(self.requests),1)

    async def test_both_channels_in_chunk_are_pending_before_await(self):
        first_graded=asyncio.Event()
        async def validate(answer):
            if answer=='3': await first_graded.wait()
        async def handler(request):
            if request.url.path=='/v1/chat/completions':
                self.requests.append(json.loads(request.content))
                return self.response_stream([tokens_chunk(r'\boxed{3}',reasoning=r'\boxed{1}')])
            response=await self.simple_handler(request)
            if request.url.path=='/verify': first_graded.set()
            return response
        result=await self.run_case(handler,validator=Validator(validate))
        self.assertEqual(result['status'],'solved',result);self.assertEqual(self.feedback,[])

    async def test_new_candidate_while_tokenizing_is_checked_before_fork(self):
        preparing=asyncio.Event(); arrived=asyncio.Event()
        class LateStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield tokens_chunk(r'\boxed{1}')
                await preparing.wait()
                yield tokens_chunk(r'\boxed{3}',count=1)
                await asyncio.Event().wait()
            async def aclose(self): pass
        async def validate(answer):
            if answer=='3': arrived.set()
        async def handler(request):
            if request.url.path=='/v1/chat/completions':
                self.requests.append(json.loads(request.content));return httpx.Response(200,stream=LateStream())
            if request.url.path=='/tokenize':
                preparing.set();await arrived.wait();await asyncio.sleep(0)
            return await self.simple_handler(request)
        result=await self.run_case(handler,validator=Validator(validate))
        self.assertEqual(result['status'],'solved',result);self.assertEqual(len(self.requests),1)
        self.assertEqual(self.checked,['1','3'])

    async def test_forks_each_active_lane_within_budget(self):
        async def handler(request):
            if request.url.path=='/v1/chat/completions':
                self.requests.append(json.loads(request.content))
                return self.response_stream([tokens_chunk(r'\boxed{1}' if len(self.requests)==1 else 'working')])
            return await self.simple_handler(request)
        result=await self.run_case(handler,changes={'rollouts':2})
        self.assertEqual(result['status'],'solved',result);self.assertEqual(len(self.requests),4)
        children=[r for r in result['rollouts'] if r['continuation_of_rollout']]
        self.assertEqual({r['continuation_of_rollout'] for r in children},{1,2})
        self.assertEqual({r['lane_id'] for r in children},{1,2})

    async def test_four_first_samples_leave_no_correction_budget(self):
        graded=asyncio.Event()
        class AfterWrong(httpx.AsyncByteStream):
            async def __aiter__(self):
                yield tokens_chunk(r'\boxed{1}')
                await graded.wait();yield tokens_chunk(r'\boxed{3}')
                await asyncio.Event().wait()
            async def aclose(self): pass
        async def handler(request):
            if request.url.path=='/v1/chat/completions':
                self.requests.append(json.loads(request.content));return httpx.Response(200,stream=AfterWrong())
            response=await self.simple_handler(request)
            if request.url.path=='/verify': graded.set()
            return response
        result=await self.run_case(handler,changes={'rollouts':4})
        self.assertEqual(result['status'],'solved',result);self.assertEqual(len(self.requests),4)
        self.assertEqual(self.feedback,[])

    async def test_unforkable_stream_is_kept_alive(self):
        for case in ('missing_ids','context_full','invalid_tokenize','continuations_disabled'):
            with self.subTest(case=case):
                output=self.output/case; output.mkdir(); self.requests=[];self.feedback=[];self.checked=[]
                graded=asyncio.Event(); second=asyncio.Event()
                class LaterCorrect(httpx.AsyncByteStream):
                    async def __aiter__(self):
                        yield tokens_chunk(r'\boxed{1}',ids=case!='missing_ids')
                        await second.wait();yield tokens_chunk(r'\boxed{3}',ids=case!='missing_ids')
                        await asyncio.Event().wait()
                    async def aclose(self): pass
                async def handler(request):
                    if request.url.path=='/v1/chat/completions':
                        self.requests.append(json.loads(request.content));return httpx.Response(200,stream=LaterCorrect())
                    if request.url.path=='/tokenize' and case=='invalid_tokenize':
                        second.set();return httpx.Response(200,json={'tokens':[True],'count':1})
                    response=await self.simple_handler(request)
                    if request.url.path=='/verify' and json.loads(request.content)['candidate']=='1':
                        graded.set()
                        if case!='invalid_tokenize': asyncio.get_running_loop().call_later(.01,second.set)
                    return response
                changes={'max_context_tokens':5} if case=='context_full' else {'no_continuation':True} if case=='continuations_disabled' else {}
                result=await self.run_case(handler,changes=changes,output=output)
                self.assertEqual(result['status'],'solved',result);self.assertEqual(len(self.requests),1)

    async def test_http_failure_is_not_a_wrong_verdict(self):
        async def handler(request):
            if request.url.path=='/verify':return httpx.Response(500,json={'error':'failed'})
            return await self.simple_handler(request)
        result=await self.run_case(handler)
        self.assertEqual(result['status'],'error');self.assertEqual(result['rejected_answers'],[])
        self.assertEqual(self.feedback,[]);self.assertTrue(all(s.closed for s in self.streams))

    async def test_wrong_after_natural_end_is_retained_in_next_fresh_round(self):
        async def handler(request):
            if request.url.path=='/v1/chat/completions':
                self.requests.append(json.loads(request.content))
                text=r'\boxed{1}' if len(self.requests)==1 else r'\boxed{3}'
                return self.response_stream([tokens_chunk(text,finish='stop'),b'data: [DONE]\n\n'],hang=False)
            return await self.simple_handler(request)
        first=await self.run_case(handler)
        self.assertEqual(first['status'],'unsolved',first)
        result=await self.run_case(handler,round_no=2,offset=1,prefixes=[None])
        self.assertEqual(result['status'],'solved',result)
        self.assertIn('grader rejected',self.requests[1]['messages'][1]['content'])
        self.assertEqual(len(result['rollouts']),2)

    async def test_cancel_during_feedback_preparation_closes_parent(self):
        preparing=asyncio.Event()
        async def handler(request):
            if request.url.path=='/tokenize':preparing.set();await asyncio.Event().wait()
            return await self.simple_handler(request)
        task=asyncio.create_task(self.run_case(handler))
        await preparing.wait();task.cancel()
        with self.assertRaises(asyncio.CancelledError):await task
        self.assertTrue(all(s.closed for s in self.streams))
        q=json.loads((self.output/'trace/01/question.json').read_text())
        self.assertEqual(len(q['rollouts']),1)

    async def test_later_barrier_resumes_latest_branch_of_each_lane(self):
        folder=self.output/'trace/01';folder.mkdir(parents=True)
        args=parse_args(['--rollouts','2'])
        records=[{'rollout':1,'round':1,'lane_id':1},
                 {'rollout':2,'round':1,'lane_id':2},
                 {'rollout':3,'round':1,'lane_id':1,'status':'completed','finish_reason':'length'}]
        records[1].update(status='completed',finish_reason='length')
        records[0].update(status='cancelled',finish_reason=None)
        (folder/'question.json').write_text(json.dumps({'round':1,'rollouts':records,'rejected_answers':['1']}))
        for r in (2,3):
            d=folder/f'rollout-{r:02d}';d.mkdir()
            (d/'tokens.json').write_text(json.dumps({'prompt_token_ids':[1], 'output_token_ids':[r],
                'complete':True,'visible_text':'working'}))
        options,round_no,used,prefixes=runner.group_options({'problem_idx':1},args,self.output)
        self.assertEqual((used,options.rollouts,round_no),(3,1,2))
        self.assertEqual(prefixes[0]['parent_rollout'],3)
        self.assertEqual(options.rejected_answers,['1'])

    async def test_only_replaceable_lane_is_cancelled_when_one_slot_remains(self):
        async def handler(request):
            if request.url.path=='/v1/chat/completions':
                self.requests.append(json.loads(request.content))
                return self.response_stream([tokens_chunk(r'\boxed{1}' if len(self.requests)==1 else 'working')])
            return await self.simple_handler(request)
        result=await self.run_case(handler,changes={'rollouts':3})
        self.assertEqual(result['status'],'solved',result)
        self.assertEqual(len(self.requests),4)
        self.assertEqual([r['continuation_of_rollout'] for r in result['rollouts']],[None,None,None,1])

    async def test_repeated_wrong_forks_never_exceed_four_and_preserve_all_feedback(self):
        async def handler(request):
            if request.url.path in ('/v1/chat/completions','/v1/completions'):
                body=json.loads(request.content);self.requests.append(body)
                n=len(self.requests)
                data=json.loads(tokens_chunk(r'\boxed{'+str(n)+'}',finish='length' if n==4 else None).decode()[6:])
                data['prompt_token_ids']=body.get('prompt',[1,2])
                return self.response_stream([('data: '+json.dumps(data)+'\n\n').encode(),
                    *([b'data: [DONE]\n\n'] if n==4 else [])],hang=n!=4)
            if request.url.path=='/verify':
                self.checked.append(json.loads(request.content)['candidate'])
                return httpx.Response(200,json={'verdict':False})
            return await self.simple_handler(request)
        result=await self.run_case(handler)
        self.assertEqual(result['status'],'unsolved',result)
        self.assertEqual(len(self.requests),4);self.assertEqual(len(result['rollouts']),4)
        self.assertEqual(result['rejected_answers'],['1','2','3','4'])
        self.assertIn('["1", "2", "3"]',self.feedback[-1]['prompt'])
        # Original prompt/output IDs survive successive feedback insertions.
        self.assertEqual(self.requests[-1]['prompt'],[1,2,10,90,91,10,90,91,10,90,91])

    async def test_finished_capped_prefix_gets_feedback_at_normal_barrier(self):
        async def handler(request):
            if request.url.path=='/v1/chat/completions':
                self.requests.append(json.loads(request.content))
                return self.response_stream([tokens_chunk(r'\boxed{1}',finish='length'),
                    b'data: [DONE]\n\n'],hang=False)
            return await self.simple_handler(request)
        first=await self.run_case(handler)
        self.assertEqual(first['status'],'unsolved',first)
        args=parse_args([])
        options,round_no,used,prefixes=runner.group_options({'problem_idx':1},args,self.output)
        result=await self.run_case(handler,round_no=round_no,offset=used,prefixes=prefixes)
        self.assertEqual(result['status'],'solved',result)
        self.assertEqual(self.requests[-1]['prompt'],[1,2,10,90,91])
        self.assertEqual(self.requests[-1]['max_tokens'],16384)
        self.assertEqual(result['rollouts'][-1]['feedback_answers'],['1'])


class ImmutableVersions(unittest.TestCase):
    def test_previous_cores_still_verify_and_prompt_parser_are_preserved(self):
        from src.common import ROOT
        from runner_final.integrity import verify_core as v1
        from runner_final.integrity_v2 import verify_core as v2
        from runner_final.integrity_v2_1 import verify_core as v21
        from runner_final.integrity_v2_2 import verify_core as v22
        from runner_final.run_frozen_v2_1 import parse_args as old_args
        for verify in (v1,v2,v21,v22):self.assertEqual(len(verify(ROOT)),64)
        self.assertEqual(parse_args([]).system_prompt,old_args([]).system_prompt)
        # The parser algorithm and its resource/fallback limits are identical;
        # only owned import paths change in the copied version.
        old=(ROOT/'runner_final/core_v2_1/syntax.py').read_text()
        new=(ROOT/'runner_final/core_v2_2/syntax.py').read_text()
        self.assertEqual(old.replace('core_v2_1','core_v2_2'),new)
