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


def combine_2024_parts(parts):
    """Validate each paper before mapping II's indices 1–15 to 16–30."""
    if len(parts) != 2:
        raise ValueError('Expected both AIME I and II 2024 papers')
    combined = []
    for offset, rows in zip((0, 15), parts):
        if any(type(r.get('problem_idx')) is not int for r in rows) or sorted(r['problem_idx'] for r in rows) != list(range(1, 16)):
            raise ValueError('Expected 15 unique local indices in each AIME 2024 paper')
        combined.extend({**row, 'problem_idx': row['problem_idx'] + offset} for row in rows)
    return normalize_rows(combined)


def download_part(client, dataset, requested_revision=None):
    import pyarrow.parquet as pq
    api = f'https://huggingface.co/api/datasets/{dataset}'
    if requested_revision:
        api += f'/revision/{requested_revision}'
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
    return rows, info, downloads


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--year', type=int, choices=YEARS, default=2025)
    parser.add_argument('--revision', help='Exact Hugging Face commit for the 2025/2026 dataset')
    parser.add_argument('--revision-i', help='Exact commit for the AIME I 2024 source')
    parser.add_argument('--revision-ii', help='Exact commit for the AIME II 2024 source')
    args = parser.parse_args(argv)
    if args.year == 2024 and args.revision:
        parser.error('AIME 2024 has two sources; use --revision-i and --revision-ii')
    if args.year != 2024 and (args.revision_i or args.revision_ii):
        parser.error('--revision-i/--revision-ii apply only to AIME 2024')
    sources, downloads = [], []
    with httpx.Client(timeout=60, follow_redirects=True) as client:
        if args.year == 2024:
            parts = []
            licenses = []
            for paper, requested_revision, offset in (('I', args.revision_i, 0), ('II', args.revision_ii, 15)):
                dataset = f'MathArena/aime_2024_{paper}'
                part, info, files = download_part(client, dataset, requested_revision)
                parts.append(part)
                downloads.extend(files)
                licenses.append(info.get('cardData', {}).get('license'))
                sources.append({'source': f'https://huggingface.co/datasets/{dataset}',
                                'revision': info['sha'], 'split': 'train', 'rows': len(part),
                                'index_offset': offset})
            rows = combine_2024_parts(parts)
            # A combined dataset has no single upstream Git commit. Hash the ordered
            # source descriptors, preserving both actual commits in the manifest.
            revision = 'composite-sha256:' + hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest()
            source = 'https://huggingface.co/MathArena'
            license_name = licenses[0] if len(set(licenses)) == 1 else None
        else:
            dataset = f'MathArena/aime_{args.year}'
            rows, info, downloads = download_part(client, dataset, args.revision)
            revision = info['sha']
            source = f'https://huggingface.co/datasets/{dataset}'
            license_name = info.get('cardData', {}).get('license')
    problems = normalize_rows(rows)
    prompts, grader, manifest_path = benchmark_paths(args.year)
    prompt_text = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in problems)
    grader_text = ''.join(json.dumps({**{k: r[k] for k in ('problem_idx', 'problem')},
                                    'answer': str(r['answer'])}, ensure_ascii=False) + '\n'
                          for r in problems)
    manifest = {'source': source,
                'revision': revision, 'split': 'train', 'rows': len(problems),
                'dataset_sha256': hashlib.sha256(prompt_text.encode()).hexdigest(),
                'grader_sha256': hashlib.sha256(grader_text.encode()).hexdigest(),
                'downloaded_at_utc': utc_now(), 'downloads': downloads,
                'license': license_name,
                'note': 'MathArena transcription and matching answers, including its contest variants; answers used only by grader.'}
    if sources:
        manifest['sources'] = sources
    for path, content in ((prompts, prompt_text), (grader, grader_text)):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix('.jsonl.tmp')
        tmp.write_text(content, encoding='utf-8')
        tmp.replace(path)
    atomic_json(manifest_path, manifest)
    print(f'Saved {len(problems)} AIME {args.year} problems and grader answers; revision {revision}')


if __name__ == '__main__':
    main()
