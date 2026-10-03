"""Regression checks for nonblocking verification, queue bounds, failed callbacks, and strict verdict validation.

Run this test module after changes to the corresponding library or runner:
    python -m unittest test.test_verify_sidecar
Tests make no inference calls.
"""
import asyncio
import json
import unittest

from src.verify_sidecar import VerificationSidecar, validate_verdict, verification_request


class VerificationSidecarTests(unittest.IsolatedAsyncioTestCase):
    async def test_pending_verification_does_not_block_stream_or_see_future(self):
        waiting, release = asyncio.Event(), asyncio.Event()
        jobs = []
        async def verify(job):
            jobs.append(job)
            waiting.set()
            await release.wait()
            return {'verdict': 'insufficient', 'reason': 'Need more proof.'}
        sidecar = VerificationSidecar('Find the sum.', verify, concurrency=1)
        try:
            sidecar.feed('reasoning', 'The answer is 70.\n', 1)
            await asyncio.wait_for(waiting.wait(), 1)
            sidecar.feed('reasoning', 'Now check more details.\n', 2)
            self.assertIn('more details', sidecar.builder.text['reasoning'])
            self.assertNotIn('more details', jobs[0]['causal_draft'])
            self.assertFalse(sidecar.finished.is_set())
        finally:
            release.set()
            await sidecar.close(drain=True)

    async def test_wrong_uncertain_and_errors_continue_until_verified(self):
        calls, approvals = [], []
        async def verify(job):
            answer = job['candidate']['answer']
            calls.append(answer)
            if answer == 73: raise ValueError('Malformed response')
            return {'verdict': {70:'rejected', 71:'insufficient', 72:'verified'}[answer], 'reason': 'Checked.'}
        async def approved(result): approvals.append(result['candidate']['answer'])
        sidecar = VerificationSidecar('Find the sum.', verify, concurrency=1, on_verified=approved)
        try:
            for i, answer in enumerate([70,71,73]):
                sidecar.feed('reasoning', f'The answer is {answer}.\n', i)
                await sidecar.queue.join()
                self.assertFalse(sidecar.finished.is_set())
            sidecar.feed('reasoning', 'The answer is 72.\n', 4)
            await sidecar.queue.join()
            self.assertEqual(approvals, [72])
            self.assertTrue(sidecar.finished.is_set())
            self.assertEqual(calls, [70,71,73,72])
        finally: await sidecar.close()

    async def test_dedup_and_queue_limit_do_not_block_producer(self):
        release = asyncio.Event()
        async def verify(job):
            await release.wait()
            return {'verdict': 'insufficient', 'reason': 'No proof.'}
        sidecar = VerificationSidecar('Find the sum.', verify, concurrency=1, pending_limit=1)
        try:
            sidecar.feed('reasoning', 'The answer is 70.\n', 1)
            sidecar.feed('reasoning', 'The answer is 70.\n', 2)
            sidecar.feed('reasoning', 'The answer is 71.\n', 3)
            self.assertEqual(sidecar.queue.qsize(), 1)
            self.assertEqual([j['answer'] for j in sidecar.dropped], [71])
            self.assertNotIn(71, sidecar.seen)
        finally:
            release.set()
            await sidecar.close(drain=True)

    async def test_shadow_verified_signal_does_not_stop_stream(self):
        async def verify(job): return {'verdict':'verified', 'reason':'Checked.'}
        sidecar = VerificationSidecar('Find the sum.', verify)
        try:
            sidecar.feed('reasoning', 'The answer is 70.\n', 1)
            await sidecar.queue.join()
            self.assertFalse(sidecar.finished.is_set())
            sidecar.feed('reasoning', 'Keep generating.\n', 2)
            self.assertIn('Keep generating', sidecar.builder.text['reasoning'])
        finally: await sidecar.close()

    async def test_failed_early_exit_callback_does_not_kill_worker(self):
        async def verify(job): return {'verdict':'verified', 'reason':'Checked.'}
        async def fail(result): raise RuntimeError('Cannot close solver')
        sidecar = VerificationSidecar('Find the sum.', verify, concurrency=1, on_verified=fail)
        try:
            sidecar.feed('reasoning', 'The answer is 70.\n', 1)
            await asyncio.wait_for(sidecar.queue.join(), 1)
            self.assertFalse(sidecar.finished.is_set())
            sidecar.feed('reasoning', 'The answer is 71.\n', 2)
            await asyncio.wait_for(sidecar.queue.join(), 1)
            self.assertEqual(len(sidecar.results), 2)
            self.assertEqual(sidecar.results[0]['callback_error'], 'RuntimeError')
        finally: await sidecar.close()


class VerificationProtocolTests(unittest.TestCase):
    def test_invalid_verdict_never_approves(self):
        for value in [{'verdict':'verified'}, {'verdict':'verified', 'reason':''},
                      {'verdict':'probably', 'reason':'Maybe'},
                      {'verdict':'verified', 'reason':'OK', 'gold':70}]:
            with self.assertRaises(ValueError): validate_verdict(value)

    def test_request_contains_candidate_and_draft_without_key(self):
        request = verification_request('test/model', 'Find the sum.', 71, 'I think 71.', 768)
        data = json.loads(request['messages'][1]['content'])
        self.assertEqual(set(data), {'original_problem','candidate_answer','unfinished_causal_draft'})
        self.assertEqual(data['candidate_answer'],71)


if __name__ == '__main__': unittest.main()
