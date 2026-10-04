"""Single-loop admission controller for dynamic sample/continuation allocation."""
import asyncio
from collections import deque
import time
from src.common import utc_now


class AllocationPool:
    def __init__(self, indices, args, artifacts, output, attempt_start):
        self.args, self.artifacts, self.output = args, artifacts, output
        self.attempt_start = attempt_start
        self.states = {q: {'used': 0, 'active': 0, 'peak_active': 0,
                           'closed': False, 'exhausted': False, 'ready': deque()}
                       for q in indices}
        self.tasks = {}
        self.changed = asyncio.Event()
        self.registered = asyncio.Event()
        self.first_submission = None
        self.halt_requested = False
        self.peak_active = 0
        self.cursor = 0
        self.admissions = []

    def register(self, index, generate, exhausted):
        self.states[index].update(generate=generate, on_exhausted=exhausted)
        if all('generate' in s for s in self.states.values()):
            self.registered.set()

    def submitted(self, index, event, when):
        if self.first_submission is None:
            self.first_submission = {'problem_idx': index, 'candidate': event['candidate'],
                                     'elapsed_s': when-self.attempt_start,
                                     'submitted_at_utc': event['verification_started_at_utc']}
            print(f"First grader submission at {when-self.attempt_start:.3f}s", flush=True)
            self.changed.set()

    def retire(self):
        for task, index in list(self.tasks.items()):
            if not task.done():
                continue
            del self.tasks[task]
            state = self.states[index]
            state['active'] -= 1
            if task.cancelled():
                if not state['closed']:
                    raise RuntimeError(f'Unexpected generation cancellation for Q{index}')
                continue
            followup = task.result()  # Propagate service/stream failures.
            if followup and not state['closed']:
                state['ready'].append(followup)

    def eligible(self, index):
        state = self.states[index]
        return (not self.halt_requested and not state['closed'] and not state['exhausted']
                and state['used'] < self.args.max_attempts_per_question)

    def choose(self):
        indices = list(self.states)
        ordered = indices[self.cursor:] + indices[:self.cursor]
        eligible = [q for q in ordered if self.eligible(q)]
        if not eligible:
            return None
        # Ready continuations first, then least active; rotate ties across questions.
        q = min(eligible, key=lambda q: (not self.states[q]['ready'], self.states[q]['active']))
        self.cursor = (indices.index(q)+1) % len(indices)
        return q

    def admit(self, index):
        state = self.states[index]
        state['used'] += 1
        rollout = state['used']
        if state['ready']:
            plan = state['ready'].popleft()
        else:
            plan = {'stage': 0, 'target_generated_tokens': (
                self.args.first_pass_max_tokens if rollout == 1 else self.args.max_tokens),
                'continuation': None}
        task = asyncio.create_task(state['generate'](rollout, plan))
        self.tasks[task] = index
        state['active'] += 1
        state['peak_active'] = max(state['peak_active'], state['active'])
        self.peak_active = max(self.peak_active, len(self.tasks))
        event = {'problem_idx': index, 'rollout': rollout, 'stage': plan['stage'],
                 'target_generated_tokens': plan['target_generated_tokens'],
                 'parent_rollout': (plan['continuation'] or {}).get('parent_rollout'),
                 'released_at_utc': utc_now(), 'elapsed_s': time.perf_counter()-self.attempt_start,
                 'active_requests': len(self.tasks), 'question_active_requests': state['active']}
        self.admissions.append(event)
        task.add_done_callback(lambda _: self.changed.set())

    async def close_question(self, index):
        state = self.states[index]
        state['closed'] = True
        state['ready'].clear()
        tasks = [t for t,q in self.tasks.items() if q == index]
        for task in tasks:
            if not task.done() and not task.cancelling():
                task.cancel()
        self.changed.set()
        await asyncio.gather(*tasks, return_exceptions=True)

    def snapshot(self):
        self.artifacts.write_json(self.output/'allocation.json', {
            'max_concurrent_requests': self.args.max_concurrent_requests,
            'peak_active_requests': self.peak_active, 'first_submission': self.first_submission,
            'initial_max_tokens': self.args.first_pass_max_tokens,
            'later_additional_max_tokens': self.args.max_tokens,
            'policy': 'eager30; ready continuations then least-active fresh starts',
            'max_requests_per_question': self.args.max_attempts_per_question,
            'questions': {str(q): {k:s[k] for k in ('used','peak_active','closed','exhausted')}
                          for q,s in self.states.items()},
            'admissions': self.admissions,
        })

    async def run(self):
        await self.registered.wait()
        try:
            while True:
                self.changed.clear()
                self.retire()
                if self.halt_requested:
                    break
                for state in self.states.values():
                    if (not state['closed'] and not state['exhausted'] and state['active'] == 0
                            and state['used'] >= self.args.max_attempts_per_question):
                        state['exhausted'] = True
                        state['on_exhausted']()
                while len(self.tasks) < self.args.max_concurrent_requests:
                    index = self.choose()
                    if index is None:
                        break
                    self.admit(index)
                if not self.tasks and all(s['closed'] or s['exhausted'] for s in self.states.values()):
                    break
                await self.changed.wait()
        finally:
            for state in self.states.values():
                state['closed'] = True
            for task in self.tasks:
                if not task.done() and not task.cancelling():
                    task.cancel()
            await asyncio.gather(*self.tasks, return_exceptions=True)
            self.snapshot()
