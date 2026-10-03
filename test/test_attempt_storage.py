"""Buffered benchmark evidence, exact continuations, and interruption persistence."""
import asyncio
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import httpx
import yaml

from src import attempt
from src.attempt_metrics import AttemptProfiler, Meter
from src.attempt_storage import AttemptArtifacts, DisabledGPUSampler
from src.attempt_runners import speedrun_v2
from test.test_attempt import Stream, chunk
from test.test_speedrun_v1 import capped


class StorageTests(unittest.TestCase):
    def test_buffer_defers_serialization_and_keeps_snapshot_and_append_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp);store = AttemptArtifacts(root, buffered=True)
            state = {'tokens': [10]};path = root / 'trace/01/question.json'
            store.write_json(path, state);state['tokens'].append(11)
            self.assertEqual(store.read_json(path), {'tokens': [10]})
            read = store.read_json(path);read['tokens'].append(12)
            self.assertEqual(store.questions(), [{'tokens': [10]}])
            log = root / 'verification.jsonl'
            with patch('src.attempt_storage.json.dumps', side_effect=AssertionError('early serialization')):
                for n in range(2):
                    with store.open_jsonl(log) as stream:
                        stream.append({'index': n})
            self.assertEqual(list(root.rglob('*')), [])
            result = store.flush()
            self.assertEqual([json.loads(s) for s in log.read_text().splitlines()], [{'index': 0}, {'index': 1}])
            self.assertEqual(result['jsonl_rows'], 2)
            self.assertEqual(store.flush()['files_written'], 0)

    def test_failed_flush_preserves_rows_without_duplicate_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AttemptArtifacts(Path(tmp), buffered=True);p = Path(tmp) / 'events.jsonl'
            p.write_text('{"old":1}\n')
            with store.open_jsonl(p) as file:file.append({'new': 2})
            with patch('pathlib.Path.replace', side_effect=OSError('disk failure')):
                with self.assertRaises(OSError):store.flush()
            self.assertEqual(p.read_text(), '{"old":1}\n')
            self.assertIn(p, store.jsonl_files)
            store.flush()
            self.assertEqual([json.loads(s) for s in p.read_text().splitlines()], [{'old':1}, {'new':2}])

    def test_gpu_file_proxy_defers_frozen_sampler_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            store=AttemptArtifacts(Path(tmp), buffered=True);p=Path(tmp)/'gpu.jsonl'
            with store.gpu_path(p).open('w') as file:
                attempt.append_json(file, {'vram_used_mib': 123})
                self.assertFalse(p.exists())
            self.assertFalse(p.exists());store.flush()
            self.assertEqual(json.loads(p.read_text())['vram_used_mib'],123)

    def test_flags_and_disabled_meter_do_not_read_clocks(self):
        for module in (attempt, speedrun_v2):
            args=module.parse_args(['--model','WeiboAI/VibeThinker-3B','--benchmark'])
            self.assertTrue(args.no_overhead_profile and args.no_gpu_telemetry and args.buffer_traces)
            defaults=module.parse_args(['--model','WeiboAI/VibeThinker-3B'])
            self.assertFalse(defaults.no_overhead_profile or defaults.no_gpu_telemetry or defaults.buffer_traces)
        with patch('src.attempt_metrics.time.perf_counter',side_effect=AssertionError()), patch('src.attempt_metrics.time.thread_time',side_effect=AssertionError()):
            with Meter(False).measure('unused'):pass


class BufferedRunnerTests(unittest.IsolatedAsyncioTestCase):
    async def test_canonical_and_speedrun_continue_exact_ids_before_any_flush(self):
        for module in (attempt, speedrun_v2):
            with self.subTest(module=module.__name__), tempfile.TemporaryDirectory() as tmp:
                root=Path(tmp);store=AttemptArtifacts(root,buffered=True)
                profiler=AttemptProfiler(root,enabled=False,artifacts=store)
                argv=['--model','WeiboAI/VibeThinker-3B','--rollouts','1','--target-correct','1','--benchmark']
                if module is attempt:argv+=['--strategy','coverage']
                args=module.parse_args(argv);requests=[]
                async def handler(request):
                    self.assertEqual(list(root.rglob('*')), [])
                    body=json.loads(request.content)
                    if request.url.path=='/verify':
                        self.assertEqual(body['candidate'],'70')
                        return httpx.Response(200,json={'verdict':True})
                    requests.append(body)
                    if len(requests)==1:
                        return httpx.Response(200,stream=Stream(capped([10],[11],'\\boxed{0')))
                    self.assertEqual(body['prompt'],[10,11])
                    return httpx.Response(200,stream=Stream(capped([10,11],[12],'70}',completion=True)))
                async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                    run=module.run_coverage if module is attempt else module.run_speedrun
                    rows=await run([{'problem_idx':1,'problem':'test'}],args,client,root,DisabledGPUSampler(),time.perf_counter(),profiler)
                self.assertEqual(rows[0]['status'],'solved');self.assertEqual(len(requests),2)
                self.assertIsNotNone(rows[0]['rollouts'][0]['ttft_s'])
                self.assertEqual(rows[0]['overhead']['timings'],{})
                self.assertEqual(list(root.rglob('*')),[])
                store.flush()
                self.assertEqual(json.loads((root/'trace/01/rollout-02/tokens.json').read_text())['prompt_token_ids'],[10,11])
                self.assertEqual(len((root/'solved.jsonl').read_text().splitlines()),1)
                self.assertGreater(len((root/'trace/01/rollout-01/stream.jsonl').read_text().splitlines()),0)

    async def test_target_cancels_all_fanout_streams_but_retains_memory_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);store=AttemptArtifacts(root,buffered=True)
            profiler=AttemptProfiler(root,enabled=False,artifacts=store)
            args=speedrun_v2.parse_args(['--model','WeiboAI/VibeThinker-3B','--benchmark','--target-correct','1'])
            streams=[];started=asyncio.Event();checks=[]
            async def handler(request):
                if request.url.path=='/verify':
                    await started.wait();checks.append(json.loads(request.content))
                    return httpx.Response(200,json={'verdict':True})
                streams.append(Stream([chunk('\\boxed{070} \\boxed{70}')],hang=True))
                if len(streams)==4:started.set()
                return httpx.Response(200,stream=streams[-1])
            async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
                rows=await speedrun_v2.run_speedrun([{'problem_idx':1,'problem':'test'}],args,client,root,DisabledGPUSampler(),time.perf_counter(),profiler)
            self.assertEqual(len(checks),1);self.assertTrue(all(s.closed for s in streams))
            self.assertTrue(all(r['status']=='cancelled' for r in rows[0]['rollouts']))
            self.assertEqual(list(root.rglob('*')),[]);store.flush()
            self.assertEqual(len(list(root.glob('trace/*/rollout-*/telemetry.json'))),4)

    async def test_managed_success_and_interrupt_flush_after_timing_and_cleanup(self):
        for module in (attempt,speedrun_v2):
            for interrupted in (False,True,"flush_failure"):
                with self.subTest(module=module.__name__,interrupted=interrupted), tempfile.TemporaryDirectory() as tmp:
                    flush_failure=interrupted=="flush_failure"
                    interrupted=interrupted is True
                    root=Path(tmp);profile=root/'models/WeiboAI/VibeThinker-3B/vllm.yaml'
                    profile.parent.mkdir(parents=True);profile.write_text(yaml.safe_dump({'max-model-len':65536,'override-generation-config':'{"max_new_tokens":16384}'}))
                    args=module.parse_args(['--model','WeiboAI/VibeThinker-3B','--benchmark','--rollouts','1','--target-correct','1','--models-dir',str(root/'models')])
                    if module is attempt:args.strategy='coverage'
                    class Process:
                        returncode=0
                        def wait(self):return 0
                    class Services:
                        def launch(self,*a,**kw):return Process()
                        async def close(self):
                            self_outer.assertEqual(list(root.glob('attempts/*/trace')),[])
                    self_outer=self
                    async def handler(request):
                        if request.url.path=='/verify':return httpx.Response(200,json={'verdict':True})
                        self.assertNotEqual(request.url.path,'/metrics')
                        return httpx.Response(200,stream=Stream([chunk('\\boxed{70}')],hang=True))
                    real_client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
                    async def interrupt(*a,**kw):
                        # Exercise finally with partial, in-memory evidence.
                        profiler=a[-1];out=a[3]
                        profiler.artifacts.write_json(out/'trace/01/question.json',{'problem_idx':1,'status':'stopped','end_to_end_latency_s':0,'unique_candidates':0,'winner':None,'round':1,'rollouts':[],'first_solved':None})
                        raise asyncio.CancelledError()
                    from unittest.mock import AsyncMock
                    health={'queries_so_far':0,'cost_c':3}
                    if hasattr(module,'dataset_provenance'):
                        health['dataset']={'sha256':module.dataset_provenance(2025)['grader_sha256']}
                    with patch.object(module,'ROOT',root), patch.object(module,'Services',Services), patch.object(module,'ensure_free'), patch.object(module,'GPUSampler',side_effect=AssertionError('GPU polling enabled')), patch.object(module,'load_questions',return_value=[{'problem_idx':1,'problem':'test'}]), patch.object(module.subprocess,'check_output',return_value='fixture'), patch.object(module,'ready',new=AsyncMock(side_effect=[{'data':[{'id':args.model,'max_model_len':65536}]},health])), patch.object(module,'warm_inference',new=AsyncMock(return_value={'latency_s':0,'batch_size':1,'tokens_per_request':32})), patch.object(module.httpx,'AsyncClient',return_value=real_client), redirect_stdout(io.StringIO()):
                        guard=patch.object(speedrun_v2,'assert_gpu_idle')
                        with guard:
                            name='run_speedrun' if module is speedrun_v2 else 'run_coverage'
                            if flush_failure:
                                with patch('src.attempt_storage.AttemptArtifacts.flush',side_effect=OSError('disk failure')),self.assertRaises(OSError):
                                    await module.run(args)
                            elif interrupted:
                                with patch.object(module,name,new=interrupt),self.assertRaises(asyncio.CancelledError):await module.run(args)
                            else:await module.run(args)
                    out=next((root/'attempts').iterdir());summary=json.loads((out/'summary.json').read_text())
                    if flush_failure:
                        self.assertEqual(summary['status'],'failed')
                        self.assertIn('Trace flush failed',summary['error'])
                        continue
                    self.assertEqual(summary['status'],'interrupted' if interrupted else 'completed')
                    self.assertTrue(summary['trace_storage']['buffered'])
                    self.assertGreater(summary['trace_storage']['files_written'],0)
                    self.assertIsNotNone(summary['official_latency_s'])
                    self.assertEqual(summary['overhead']['engine']['samples'],0)
                    self.assertTrue((out/'trace/01/question.json').exists())
                    self.assertFalse((out/'gpu.jsonl').exists())
