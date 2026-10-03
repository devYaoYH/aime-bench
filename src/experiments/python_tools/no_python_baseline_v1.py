"""Matched OpenRouter no-Python pass@2 baseline from saved tool-arm requests.

Preserves the historical first request for each question/sample except tools,
tool_choice, and the tool-specific system suffix. Never executes model code.
Run locally: python -m src.experiments.python_tools.no_python_baseline_v1
"""
from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import subprocess
import time

import httpx

from src.common import ROOT, OPENROUTER_URL, atomic_json, extract_answer, load_key, utc_now
from src.python_tool_protocol import SYSTEM, totals

RUNNER_VERSION = 'no_python_baseline_v1'


def baseline_request(reference):
    request = deepcopy(reference['rounds'][0]['request'])
    if request['messages'][0]['role'] != 'system' or not request['messages'][0]['content'].startswith(SYSTEM):
        raise ValueError('Historical system prompt does not match the shared solver prompt')
    request.pop('tools')
    request.pop('tool_choice')
    request['messages'][0]['content'] = SYSTEM
    return request


def prepare(reference_dir):
    original = json.loads((reference_dir / 'config.json').read_text())
    if original['endpoint'] != OPENROUTER_URL or original['provider'] != 'parasail':
        raise ValueError('Expected the historical OpenRouter/Parasail experiment')
    if original['model'] != 'qwen/qwen3.5-35b-a3b' or original['samples'] != 2:
        raise ValueError('Expected Qwen3.5-35B-A3B pass@2')
    references = {}
    hashes = {}
    for s in range(1, original['samples'] + 1):
        for q in original['questions']:
            path = reference_dir / f'{q:02d}-sample-{s}.json'
            record = json.loads(path.read_text())
            if record['status'] == 'running' or record['problem_idx'] != q or record['sample_idx'] != s:
                raise ValueError(f'Invalid historical record: {path.name}')
            request = baseline_request(record)
            expected = {'model': original['model'], 'sampling': original['sampling'],
                        'seed': original['seed_base'] + 100*q + s,
                        'budget': original['total_generation_budget']}
            if (request['model'] != expected['model'] or request['seed'] != expected['seed']
                    or request['max_tokens'] != expected['budget']
                    or any(request[k] != v for k, v in expected['sampling'].items())
                    or request['provider'] != {'order': ['parasail'], 'allow_fallbacks': False, 'require_parameters': True}
                    or request['messages'] != [{'role': 'system', 'content': SYSTEM},
                                                {'role': 'user', 'content': record['problem']}]
                    or any(r['response'].get('provider') != 'Parasail' for r in record['rounds'])):
                raise ValueError(f'Historical control mismatch: {path.name}')
            references[q, s] = record
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    config = {k: deepcopy(original[k]) for k in ['model', 'provider', 'endpoint', 'source_run', 'questions',
              'samples', 'sampling', 'seed_base', 'total_generation_budget', 'concurrency', 'launch_order',
              'key_in_model_inputs', 'automatic_retries']}
    config.update(runner_version=RUNNER_VERSION, reference_run=str(reference_dir.relative_to(ROOT)),
                  reference_config_sha256=hashlib.sha256((reference_dir/'config.json').read_bytes()).hexdigest(),
                  reference_record_sha256=hashes, system=SYSTEM, tools_enabled=False,
                  request_changes=['remove tools', 'remove tool_choice', 'remove tool-specific system suffix'],
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    return config, references


async def baseline_case(client, key, reference, config, path):
    request = baseline_request(reference)
    q, sample = reference['problem_idx'], reference['sample_idx']
    record = {'problem_idx': q, 'sample_idx': sample, 'model': config['model'],
              'problem': reference['problem'], 'status': 'running', 'candidate': None,
              'started_at_utc': utc_now(), 'rounds': [], 'tool_executions': [], 'first_tool': None}
    item = {'number': 1, 'request': request, 'started_at_utc': utc_now()}
    record['rounds'].append(item)
    atomic_json(path, record)
    start = time.perf_counter()
    print(f'START Q{q:02d}/{sample}', flush=True)
    try:
        response = await client.post(OPENROUTER_URL,
            headers={'Authorization': f'Bearer {key}', 'X-Title': 'AIME Qwen3.5 matched no-Python pass@2'},
            json=request)
        item['http_status'] = response.status_code
        try:
            item['response'] = response.json()
        except ValueError:
            item['response'] = {'raw_text': response.text}
        item['latency_s'] = time.perf_counter() - start
        response.raise_for_status()
        body = item['response']
        if body.get('error'):
            raise ValueError(f'API error: {body["error"]}')
        if body.get('provider') != 'Parasail' or body.get('model') != config['model']:
            raise ValueError('Response provider/model differs from frozen historical control')
        tokens = (body.get('usage') or {}).get('completion_tokens')
        if type(tokens) is not int or not 0 <= tokens <= request['max_tokens']:
            raise ValueError('Missing or invalid completion usage')
        choice = body['choices'][0]
        message = choice['message']
        if message.get('tool_calls'):
            raise ValueError('Unexpected tool calls in no-tool arm; no code executed')
        record['final_content'] = message.get('content')
        record['candidate'] = extract_answer(record['final_content'])
        record['status'] = ('generation_budget_exhausted' if choice['finish_reason'] == 'length'
                            else 'complete' if choice['finish_reason'] == 'stop' else 'unexpected_finish')
    except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
        item['latency_s'] = time.perf_counter() - start
        item['error'] = f'{type(exc).__name__}: {exc}'
        record['status'] = 'error'
    record['finished_at_utc'] = utc_now()
    record['elapsed_s'] = time.perf_counter() - start
    record['accounting'] = totals(record['rounds'])
    record['accounting'].update(tool_calls=0, successful_tool_calls=0, tool_wall_s=0)
    record['gold_answer'] = reference['gold_answer']
    record['answer_correct'] = record['candidate'] is not None and int(record['candidate']) == record['gold_answer']
    record['correct'] = record['status'] == 'complete' and record['answer_correct']
    atomic_json(path, record)
    print(f'END Q{q:02d}/{sample}: {record["status"]}, correct={record["correct"]}, '
          f'tokens={record["accounting"]["completion_tokens"]}, {record["elapsed_s"]:.1f}s', flush=True)
    return record


def summary(records, config, wall_s):
    return {'config': config, 'profile_wall_s': wall_s, 'attempts_observed': len(records),
            'all_attempts_observed': len(records) == len(config['questions'])*config['samples'],
            'questions_correct_pass_at_2': len({r['problem_idx'] for r in records if r['correct']}),
            'attempts_correct': sum(r['correct'] for r in records),
            'status_counts': dict(Counter(r['status'] for r in records)),
            'response_providers': dict(Counter(r.get('response', {}).get('provider', 'missing')
                                              for case in records for r in case['rounds'])),
            'total_completion_tokens': sum(r['accounting']['completion_tokens'] for r in records)
                if all(r['accounting']['completion_tokens'] is not None for r in records) else None,
            'total_prompt_tokens': sum(r['accounting']['prompt_tokens'] or 0 for r in records),
            'reported_cost': sum(r['accounting']['reported_cost'] for r in records)}


async def run(args):
    reference_dir, output = (ROOT/args.reference).resolve(), (ROOT/args.out).resolve()
    if not all(p.is_relative_to(ROOT/'runs') for p in [reference_dir, output]) or output == reference_dir:
        raise ValueError('Use a distinct output under runs/')
    config, references = prepare(reference_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output/'config.json'
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError('Configuration changed; use a fresh output label')
    atomic_json(config_path, config)
    if args.prepare_only:
        print(json.dumps({k:v for k,v in config.items() if k != 'reference_record_sha256'}, indent=2))
        return
    records = []
    pending = []
    for pair, reference in references.items():
        path = output/f'{pair[0]:02d}-sample-{pair[1]}.json'
        if path.exists():
            saved = json.loads(path.read_text())
            if saved['status'] == 'running':
                raise ValueError(f'Unfinished paid request: {path.name}; inspect before restarting')
            records.append(saved)
        else:
            pending.append((reference, path))
    if not pending:
        print('CACHED: all terminal records exist; original summary preserved')
        return
    if records:
        raise ValueError('Partial saved run; use a fresh label or inspect before issuing more paid calls')
    atomic_json(output/'provenance.json', {'started_at_utc': utc_now(),
        'git_commit': subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip(),
        'command': ['python', '-m', 'src.experiments.python_tools.no_python_baseline_v1',
                    '--reference', args.reference, '--out', args.out],
        'execution_host': 'local client; hosted inference via OpenRouter only',
        'provider_documentation': 'https://openrouter.ai/docs/guides/routing/provider-selection'})
    key = load_key()
    slots = asyncio.Semaphore(config['concurrency'])
    start = time.perf_counter()
    async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=30),
            limits=httpx.Limits(max_connections=config['concurrency']+2)) as client:
        async def one(reference, path):
            async with slots:
                record = await baseline_case(client, key, reference, config, path)
            records.append(record)
            result = summary(records, config, time.perf_counter()-start)
            atomic_json(output/'summary.json', result)
            print(f'PROGRESS {len(records)}/60; pass@2={result["questions_correct_pass_at_2"]}/30', flush=True)
        await asyncio.gather(*(one(reference, path) for reference, path in pending))
    result = summary(records, config, time.perf_counter()-start)
    atomic_json(output/'summary.json', result)
    print(json.dumps({k:v for k,v in result.items() if k != 'config'}, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', default='runs/python-tool-qwen35-pass2-auto')
    p.add_argument('--out', default='runs/no-python-qwen35-pass2-parasail-20261003')
    p.add_argument('--prepare-only', action='store_true')
    asyncio.run(run(p.parse_args()))
