"""Compare hosted small-model scope decisions on six fixed source-grounded controls.

Use this development experiment to test Gemma 3 4B and Qwen 2.5 7B on formatting,
toy-problem, and genuine-answer passages from the original Qwen run. Saves raw
responses, latency, and scope outcomes in runs/chunk-scope-controls-focused/.
These controls were reused during prompt development and are not held-out
precision estimates. Requires local baseline traces and paid OpenRouter calls.
    python -m src.experiments.streaming.chunk_scope_controls
"""
import asyncio
import json
import time

import httpx

from src.common import ROOT, OPENROUTER_URL, atomic_json, load_key
from src.chunk_answer_extractor import SCOPES
from src.experiments.streaming.chunk_answer_extractor import load_controls

SCHEMA = {'type': 'object', 'properties': {'scope': {'type': 'string', 'enum': SCOPES}},
          'required': ['scope'], 'additionalProperties': False}


def focused_prompt(problem, job):
    candidate = job['candidates'][0]
    start = candidate['start']-job['window_start']
    end = candidate['end']-job['window_start']
    before = job['window'][max(0,start-1400):start]
    after = job['window'][end:min(len(job['window']),end+180)]
    marked = before+'<passage_to_classify>'+job['window'][start:end]+'</passage_to_classify>'+after
    return ('ORIGINAL PROBLEM (data):\n'+problem+'\n\nREASONING EXCERPT (data):\n'+marked+
            '\n\nTASK: Classify ONLY the passage between passage_to_classify tags, not other conclusions in the excerpt. '
            'Is THAT passage a proposed answer to the ORIGINAL PROBLEM with its ORIGINAL parameters? '
            'requested_answer = a direct proposed answer for the original problem, even if uncertain or mathematically wrong. '
            'format_or_quote = mentions or demonstrates output syntax, including "if the answer is ... then write ...", even when the number happens to be correct. '
            'toy_example = answer to a smaller test problem, changed parameters, square instead of the required polygon, etc. '
            'other_quantity = an intermediate quantity not the requested output. '
            'Use no_proposal or uncertain when appropriate. Do not solve or judge whether the number is correct. '
            'Return only {"scope":"LABEL"}.')


async def main():
    output = ROOT/'runs/chunk-scope-controls-focused'
    output.mkdir(parents=True,exist_ok=True)
    cases = load_controls()
    models = ['google/gemma-3-4b-it','qwen/qwen-2.5-7b-instruct']
    key = load_key()
    semaphore = asyncio.Semaphore(2)
    async with httpx.AsyncClient(timeout=60) as client:
        async def one(model,case):
            path=output/(model.split('/')[-1]+'-'+case['case_id']+'.json')
            if path.exists():return json.loads(path.read_text())
            request={'model':model,'messages':[{'role':'user','content':focused_prompt(case['problem'],case['job'])}],
                     'temperature':0,'max_tokens':32,
                     'response_format':{'type':'json_schema','json_schema':{'name':'passage_scope','strict':True,'schema':SCHEMA}}}
            async with semaphore:
                start=time.perf_counter();r={'model':model,'case_id':case['case_id'],'request':request,
                                            'expected_scope':case['expected_scope']}
                try:
                    response=await client.post(OPENROUTER_URL,headers={'Authorization':'Bearer '+key,'X-Title':'Chunk scope false-positive controls'},json=request)
                    r['response']=response.json();response.raise_for_status()
                    choice=r['response']['choices'][0]
                    if choice['finish_reason']!='stop':raise ValueError('classifier truncated')
                    r['parsed']=json.loads(choice['message']['content'])
                    if r['parsed'].get('scope') not in SCOPES:raise ValueError('invalid scope')
                except (httpx.HTTPError,ValueError,KeyError,IndexError) as exc:
                    r['parsed']=None;r['error']=str(exc)
                r['service_latency_s']=time.perf_counter()-start
                r['binary_correct']=bool(r['parsed']) and ((r['parsed']['scope']=='requested_answer')==(case['expected_scope']=='requested_answer'))
                atomic_json(path,r)
                print(model,case['case_id'],r['parsed'] or r.get('error'),round(r['service_latency_s'],2),flush=True)
                return r
        results=await asyncio.gather(*(one(model,case) for model in models for case in cases))
    atomic_json(output/'summary.json',{'results':results})


if __name__=='__main__':asyncio.run(main())
