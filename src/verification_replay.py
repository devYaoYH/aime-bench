"""Simulate shared verification queues and generation-slot reuse from saved candidates.

Import candidate_schedule and simulate to compare final-only, marker, and
permissive checking under the same service cost and trajectory durations.
Wrong candidates consume verifier time; keys are consulted only at check
completion. Timing and cancellation are counterfactual assumptions, not measured
solver behavior. The original-Qwen and Python-tool back-tests share this library.
"""
from collections import deque
import heapq
import itertools

def candidate_schedule(row, policy, *, exponent=1, first_token_s=0):
    duration = row['api_latency_s']
    if duration <= 0 or exponent <= 0 or not 0 <= first_token_s < duration:
        raise ValueError('Invalid timing assumptions')
    events = []
    if policy != 'final':
        for e in row['candidates']:
            if policy=='markers' and e['kind'] not in ['boxed','answer_line']: continue
            if 'segment_start_s' in e:
                # Multi-round traces: vary delivery only inside this API round.
                # CPU tool time and earlier rounds must remain before the candidate.
                segment_duration=e['segment_duration_s']
                delay=min(first_token_s,max(0,segment_duration-1e-6))
                t=e['segment_start_s']+delay+(segment_duration-delay)*e['segment_fraction']**exponent
            elif 'offset_s' in e:
                # Tool results become available after execution, not during decode.
                t=e['offset_s']
            else:
                t = first_token_s+(duration-first_token_s)*e['output_fraction']**exponent
            events.append({'offset_s':t, 'answer':e['answer'], 'kind':e['kind'],
                           'confidence':e['confidence'], 'quote':e['quote'],
                           'output_fraction':e['output_fraction'],
                           **{k:e[k] for k in ['round_number','part','source_file'] if k in e}})
    # Final-only baseline and permissive policy both retain the natural endpoint.
    if row['final_candidate'] is not None:
        events.append({'offset_s':duration, 'answer':row['final_candidate'],
                       'kind':'final', 'quote':'saved final response'})
    return sorted(events, key=lambda e:e['offset_s'])

def simulate(rows, *, policy='permissive', generation_slots=120, service_s=3,
             exponent=1, first_token_s=0, stop_on_verified=True):
    """FIFO generation and verification; keys consulted only at verifier finish.

    Question-answer deduplication precedes grading. Wrong answers consume full
    service time. Solved questions retire their queued checks; the current job
    is never preempted. With 120 slots all selected trajectories start at t=0.
    """
    if service_s <= 0 or generation_slots < 1: raise ValueError('Positive capacity required')
    ordered = sorted(rows, key=lambda r:(r['sample_number'],r['problem_idx']))
    pending = deque(ordered)
    active, canceled, seen, solved = {}, set(), set(), {}
    checks, verification_queue, heap, starts = [], deque(), [], []
    serial = itertools.count()
    busy = False
    now = 0.0
    peak_queue = 0
    discarded_checks = 0

    def push(t, kind, payload):
        priority = {'verify_done':0,'candidate':1,'generation_end':2}[kind]
        heapq.heappush(heap,(t,priority,next(serial),kind,payload))

    def fill(t):
        while len(active) < generation_slots and pending:
            row = pending.popleft()
            if stop_on_verified and row['problem_idx'] in solved: continue
            tid = (row['problem_idx'],row['sample_number'])
            active[tid] = row
            starts.append({'problem_idx':tid[0], 'sample_number':tid[1], 'start_s':t,
                           'natural_end_s':t+row['api_latency_s']})
            push(t+row['api_latency_s'],'generation_end',tid)
            for e in candidate_schedule(row,policy,exponent=exponent,first_token_s=first_token_s):
                push(t+e['offset_s'],'candidate',{'trace_id':tid, 'problem_idx':tid[0],
                    'sample_number':tid[1], 'gold':row['gold_answer'], 'arrival_s':t+e['offset_s'], **e})

    def start_check(t):
        nonlocal busy, discarded_checks
        if busy: return
        while verification_queue:
            job = verification_queue.popleft()
            if job['problem_idx'] in solved:
                discarded_checks += 1
                continue
            busy = True
            push(t+service_s,'verify_done',{**job, 'verification_start_s':t,
                 'verified_at_s':t+service_s, 'queue_s':t-job['arrival_s']})
            return

    fill(now)
    while heap:
        now,_,_,kind,data = heapq.heappop(heap)
        if kind=='candidate':
            q = data['problem_idx']
            key = (q,data['answer'])
            if data['trace_id'] in canceled or q in solved or key in seen: continue
            seen.add(key)
            verification_queue.append(data)
            peak_queue = max(peak_queue,len(verification_queue))
        elif kind=='generation_end':
            if data not in active: continue
            del active[data]
            fill(now)
        elif kind=='verify_done':
            busy = False
            correct = data['answer']==data['gold']
            # The only correctness branch occurs after a charged verification.
            data['correct'] = correct
            checks.append(data)
            if correct and data['problem_idx'] not in solved:
                q = data['problem_idx']
                solved[q] = {'problem_idx':q, 'time_s':now, 'answer':data['answer'],
                             'sample_number':data['sample_number'], 'kind':data['kind'],
                             'queue_s':data['queue_s']}
                if stop_on_verified:
                    for tid in list(active):
                        if tid[0]==q:
                            canceled.add(tid)
                            del active[tid]
                    fill(now)
        start_check(now)
    milestones = sorted(solved.values(),key=lambda r:(r['time_s'],r['problem_idx']))
    for n,row in enumerate(milestones,1): row['n_correct']=n
    return {'policy':policy, 'generation_slots':generation_slots,
            'service_s':service_s, 'exponent':exponent, 'first_token_s':first_token_s,
            'stop_on_verified':stop_on_verified, 'correct_questions':len(milestones),
            'time_to_18_s':milestones[17]['time_s'] if len(milestones)>=18 else None,
            'checks_completed':len(checks), 'wrong_checks':sum(not c['correct'] for c in checks),
            'checks_before_18':sum(c['verified_at_s']<=milestones[17]['time_s'] for c in checks) if len(milestones)>=18 else None,
            'peak_pending_checks':peak_queue, 'discarded_solved_checks':discarded_checks,
            'trajectories_started':len(starts), 'canceled_trajectories':len(canceled),
            'milestones':milestones, 'checks':checks, 'generation_starts':starts}
