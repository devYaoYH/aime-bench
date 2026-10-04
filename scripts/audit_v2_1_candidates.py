"""Replay historical v2 client candidate/verdict evidence through the v2.1 gate.

No dataset, gold field, SSE stream, grader audit or model service is read.
This cannot establish a new GPU timing result or simulate changed scheduling.
"""
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

from runner_final.core_v2_1.syntax import ExpressionValidator, POLICY
from runner_final.integrity_v2_1 import verify_core
from src.common import ROOT, atomic_json

DEFAULT_BATCH = ROOT/'runs/experiments/core-v2-aime2025-five-seeds-20261004T013100Z'
DEFAULT_OUTPUT = ROOT/'runs/analyses/core-v2_1-candidate-validation'


async def audit(batch, output):
    summary = json.loads((batch/'summary.json').read_text())
    rows, files = [], {}
    async with ExpressionValidator() as validator:
        for trial in summary['trials']:
            folder = ROOT/'attempts'/trial['attempt_id']
            for path in sorted(folder.glob('trace/*/verification.jsonl')):
                files[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
                seen = set()
                for line in path.read_text().splitlines():
                    event = json.loads(line)
                    verdict = event.get('result', {}).get('verdict')
                    if type(verdict) is not bool:
                        continue
                    result = await validator.validate(event['candidate'])
                    key = result.get('canonical_key')
                    duplicate = result['valid'] and key in seen
                    if result['valid']:
                        seen.add(key)
                    rows.append({'attempt_id':folder.name, 'problem_idx':int(path.parent.name),
                                 'candidate':event['candidate'], 'recorded_verdict':verdict,
                                 'would_suppress_equivalent':bool(duplicate), **result})
    correct = [r for r in rows if r['recorded_verdict']]
    wrong = [r for r in rows if not r['recorded_verdict']]
    uncached = [r for r in rows if not r['cache_hit']]
    result = {'core_manifest_sha256':verify_core(ROOT), 'policy':POLICY,
              'completed_at_utc':datetime.now(timezone.utc).isoformat(),
              'reference_batch':str(batch.relative_to(ROOT)), 'input_sha256':files,
              'recorded_checks':len(rows), 'recorded_correct':len(correct),
              'correct_syntax_retained':sum(r['valid'] for r in correct),
              'wrong_syntax_rejected':sum(not r['valid'] for r in wrong),
              'placeholder_rejected':sum(r.get('reason')=='placeholder' for r in wrong),
              'equivalent_duplicates_suppressed':sum(r['would_suppress_equivalent'] for r in rows),
              'validation_wall_median_ms':statistics.median(r['validation_wall_s'] for r in rows)*1000,
              'validation_wall_max_ms':max(r['validation_wall_s'] for r in rows)*1000,
              'uncached_proposals':len(uncached),
              'uncached_wall_median_ms':statistics.median(r['validation_wall_s'] for r in uncached)*1000,
              'uncached_wall_max_ms':max(r['validation_wall_s'] for r in uncached)*1000,
              'scope':'Offline replay on local CPU, cached and sequential. Grader client verdicts only. No GPU run or counterfactual time saved is claimed.'}
    output.mkdir(parents=True,exist_ok=True)
    atomic_json(output/'summary.json',result)
    atomic_json(output/'candidate_audit.json',rows)
    print(json.dumps(result,indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch',type=Path,default=DEFAULT_BATCH)
    parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT)
    args=parser.parse_args()
    asyncio.run(audit(args.batch.resolve(),args.output.resolve()))


if __name__=='__main__':
    main()
