"""Build native Python-tool conversation history and cumulative usage accounting.

Use these helpers in multi-round OpenRouter runners to preserve tool-call IDs
and reasoning fields, validate python_math arguments, and sum tokens, cost, and
latency across rounds. The paired pilot and pass@2 profile share the concise
solver prompt here. Importing the module does not execute generated Python.
"""
from copy import deepcopy
import json

SYSTEM=('Solve the mathematical problem carefully using exact reasoning. '
        'Keep the derivation concise; finish once the requested answer is established. '
        'End with a standalone line Answer: NNN, where NNN is an integer from 0 to 999.')

def assistant_history(message):
    # Preserve interleaved reasoning fields and exact call IDs on follow-up turns.
    return {k:deepcopy(message[k]) for k in ['role','content','tool_calls','reasoning','reasoning_details'] if k in message}


def parse_tool_call(call):
    if call.get('type')!='function' or call.get('function',{}).get('name')!='python_math':
        raise ValueError('Unknown tool; only python_math is available')
    if not isinstance(call.get('id'),str) or not call['id']: raise ValueError('Missing tool call ID')
    args=json.loads(call['function']['arguments'])
    if not isinstance(args,dict) or set(args)!={'code'} or not isinstance(args['code'],str):
        raise ValueError('Tool arguments must contain only the code string')
    return args['code']


def totals(rounds):
    usage=[(r.get('response') or {}).get('usage') or {} for r in rounds]
    result={name:sum(u[name] for u in usage) if usage and all(isinstance(u.get(name),int) for u in usage) else None
            for name in ['prompt_tokens','completion_tokens','total_tokens']}
    reasons=[(u.get('completion_tokens_details') or {}).get('reasoning_tokens') for u in usage]
    result['reasoning_tokens']=sum(reasons) if reasons and all(isinstance(n,int) for n in reasons) else None
    result['reported_cost']=sum(u.get('cost') or 0 for u in usage)
    result['rounds_with_missing_usage']=sum(not isinstance(u.get('completion_tokens'),int) for u in usage)
    result['api_rounds']=len(rounds)
    result['api_latency_sum_s']=sum(r.get('latency_s',0) for r in rounds)
    return result
