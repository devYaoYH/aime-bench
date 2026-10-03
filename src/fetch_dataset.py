"""Download a revision-pinned MathArena AIME dataset into prompts and grader.

Run `python -m src.fetch_dataset --year 2026`. The default remains 2025.
Both JSONL files and a source manifest are generated from the same pinned
Parquet file. No inference occurs. Only this downloader needs pyarrow.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json

import httpx

from src.benchmarks import YEARS, benchmark_paths
from src.common import atomic_json, utc_now


def normalize_rows(rows):
    problems = []
    for row in rows:
        if type(row.get('problem_idx')) is not int or type(row.get('answer')) is not int:
            raise ValueError('AIME indices and answers must be integers')
        if not 0 <= row['answer'] <= 999:
            raise ValueError('AIME answers must lie between 0 and 999')
        if not isinstance(row.get('problem'), str) or not row['problem'].strip():
            raise ValueError('Empty AIME problem statement')
        problems.append({key: row[key] for key in ('problem_idx', 'problem', 'answer')}
                        | {'problem_type': row.get('problem_type', [])})
    problems.sort(key=lambda row: row['problem_idx'])
    if [r['problem_idx'] for r in problems] != list(range(1, 31)):
        raise ValueError('Expected exactly 30 unique AIME problem indices')
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--year', type=int, choices=YEARS, default=2025)
    parser.add_argument('--revision', help='Optional exact Hugging Face commit to download')
    args = parser.parse_args(argv)
    import pyarrow.parquet as pq

    dataset = f'MathArena/aime_{args.year}'
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        api = f'https://huggingface.co/api/datasets/{dataset}'
        if args.revision:
            api += f'/revision/{args.revision}'
        info = client.get(api).raise_for_status().json()
        revision = info['sha']
        files = sorted(f['rfilename'] for f in info['siblings']
                       if f['rfilename'].startswith('data/train-') and f['rfilename'].endswith('.parquet'))
        if not files:
            raise ValueError('No train Parquet files in the pinned dataset')
        rows, downloads = [], []
        for filename in files:
            url = f'https://huggingface.co/datasets/{dataset}/resolve/{revision}/{filename}'
            content = client.get(url).raise_for_status().content
            rows.extend(pq.read_table(io.BytesIO(content)).to_pylist())
            downloads.append({'url': url, 'sha256': hashlib.sha256(content).hexdigest()})
    problems = normalize_rows(rows)
    prompts, grader, manifest_path = benchmark_paths(args.year)
    prompt_text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in problems)
    grader_text = ''.join(json.dumps({**{k: r[k] for k in ('problem_idx', 'problem')},
                                    'answer': str(r['answer'])}, ensure_ascii=False) + '\n'
                          for r in problems)
    manifest = {'source': f'https://huggingface.co/datasets/{dataset}',
                'revision': revision, 'split': 'train', 'rows': len(problems),
                'dataset_sha256': hashlib.sha256(prompt_text.encode()).hexdigest(),
                'grader_sha256': hashlib.sha256(grader_text.encode()).hexdigest(),
                'downloaded_at_utc': utc_now(), 'downloads': downloads,
                'license': info.get('cardData', {}).get('license'),
                'note': 'MathArena transcription and matching answers, including its contest variants; answers used only by grader.'}
    for path, content in ((prompts, prompt_text), (grader, grader_text)):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.jsonl.tmp')
        tmp.write_text(content, encoding='utf-8')
        tmp.replace(path)
    atomic_json(manifest_path, manifest)
    print(f'Saved {len(problems)} AIME {args.year} problems and grader answers; revision {revision}')


if __name__ == '__main__':
    main()
