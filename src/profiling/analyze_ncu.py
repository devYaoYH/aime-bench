"""Derive hardware Tensor Core roofline points from exported Nsight counters."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from src.common import ROOT, atomic_json

TENSOR = 'sm__ops_path_tensor_src_bf16_dst_fp32_sparsity_off'


def value(row, name):
    text = row.get(name, '')
    if text in ('', 'n/a', 'N/A'):
        return None
    return float(text.replace(',', ''))


def roofline(row, units):
    duration = value(row, 'gpu__time_duration.sum')
    scale = {'ns': 1e-9, 'us': 1e-6, 'µs': 1e-6, 'ms': 1e-3, 's': 1}
    if units['gpu__time_duration.sum'] not in scale:
        raise ValueError('Unknown duration unit')
    seconds = duration * scale[units['gpu__time_duration.sum']]
    ops = value(row, TENSOR+'.sum')
    traffic = value(row, 'dram__bytes.sum')
    if traffic is None:
        traffic = value(row, 'dram__bytes_read.sum') + value(row, 'dram__bytes_write.sum')
    if not seconds or not traffic or ops is None:
        raise ValueError('Missing measured roofline inputs')
    intensity = ops / traffic
    performance = ops / seconds
    bw = traffic / seconds
    nominal_roof = min(312e12, intensity * 1935e9)
    return {'duration_us': seconds*1e6, 'bf16_dense_tensor_ops': ops,
            'dram_bytes': traffic, 'tensor_ops_per_dram_byte': intensity,
            'tensor_tops': performance/1e12, 'dram_gb_per_s': bw/1e9,
            'nominal_hbm_roof_pct': bw/1935e9*100,
            'nominal_dense_tensor_roof_pct': performance/312e12*100,
            'nominal_roof_at_intensity_pct': performance/nominal_roof*100 if nominal_roof else None,
            'dram_counter_pct': value(row, 'dram__throughput.avg.pct_of_peak_sustained_elapsed'),
            'sm_counter_pct': value(row, 'sm__throughput.avg.pct_of_peak_sustained_elapsed'),
            'tensor_pipe_pct': value(row, 'sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed'),
            'l2_hit_pct': value(row, 'lts__t_sector_hit_rate.pct'),
            'l2_bytes': value(row, 'lts__t_bytes.sum'),
            'active_warps_pct': value(row, 'sm__warps_active.avg.pct_of_peak_sustained_active'),
            'issue_active_pct': value(row, 'smsp__issue_active.avg.pct_of_peak_sustained_active')}


def analyze(folder):
    with (folder/'ncu_raw.csv').open() as stream:
        lines = [line for line in stream if not line.startswith('==')]
    rows = list(csv.DictReader(lines))
    units, data = rows[0], rows[1:]
    measured = []
    for row in data:
        point = {'id': int(row['ID']), 'name': row['Kernel Name'], **roofline(row, units)}
        measured.append(point)
    summary = json.loads((folder/'summary.json').read_text())
    windows = summary['windows']
    # CUDA profiler start/stop markers aren't in raw CSV. Persist the observed
    # graph IDs assigned to each window separately in window_metrics.json.
    mapping_path = folder/'window_metrics.json'
    mapping = json.loads(mapping_path.read_text()) if mapping_path.exists() else {}
    for point in measured:
        point['phase'] = mapping.get(str(point['id']))
    attempt = ROOT/'attempts'/summary['attempt_id']
    config = json.loads((attempt/'config.json').read_text())
    control = ROOT/'attempts/20261004T005518.974361Z'
    differences = []
    for index in range(1, 31):
        path = Path('trace')/f'{index:02d}'/'rollout-01/request.json'
        if json.loads((attempt/path).read_text()) != json.loads((control/path).read_text()):
            differences.append(index)
    q = [json.loads(p.read_text()) for p in sorted(attempt.glob('trace/*/question.json'))]
    rollouts = [r for row in q for r in row['rollouts']]
    controls = json.loads((control/'config.json').read_text())
    keys = ['runner_id', 'core_manifest_sha256', 'system_prompt_sha256', 'seed',
            'model_profile_sha256', 'temperature', 'top_p', 'parallelism', 'rollouts',
            'first_pass_max_tokens', 'max_tokens', 'max_attempts_per_question',
            'schedule', 'target_correct', 'grader_cost', 'benchmark', 'buffer_traces',
            'no_gpu_telemetry', 'no_overhead_profile', 'skip_benchmark_prewarm']
    result = {'scope': 'Selected workloads in one instrumented attempt; not a workload-frequency or scored-time estimate.',
              'raw_csv_sha256': hashlib.sha256((folder/'ncu_raw.csv').read_bytes()).hexdigest(),
              'profiled_workloads': len(measured), 'nonzero_tensor_workloads': sum(p['bf16_dense_tensor_ops'] > 0 for p in measured),
              'points': measured, 'windows': windows,
              'control': {'reference_attempt_id': control.name, 'different_initial_questions': differences,
                          'different_control_fields': [k for k in keys if config.get(k) != controls.get(k)],
                          'all_question_request_caps_valid': all(len(row['rollouts']) <= 4 for row in q),
                          'generation_requests': len(rollouts),
              'continuations': sum(r['continuation_of_rollout'] is not None for r in rollouts),
              'later_fresh': sum(r['rollout'] > 1 and r['continuation_of_rollout'] is None for r in rollouts)}}
    with (folder/'serving_samples.csv').open() as stream:
        serving = list(csv.DictReader(stream))
    result['serving_context'] = {
        'sample_count': len(serving),
        'max_observed_kv_fraction': max(float(r['vllm:kv_cache_usage_perc']) for r in serving),
        'max_waiting_requests': max(float(r['vllm:num_requests_waiting']) for r in serving),
        'samples_with_waiting_requests': sum(float(r['vllm:num_requests_waiting']) > 0 for r in serving),
        'max_running_requests': max(float(r['vllm:num_requests_running']) for r in serving)}
    atomic_json(folder/'analysis.json', result)
    with (folder/'roofline_points.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(measured[0]), lineterminator='\n')
        writer.writeheader(); writer.writerows(measured)
    return result


def plot(folder, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    points = [p for p in result['points'] if p['bf16_dense_tensor_ops'] > 0]
    fig, ax = plt.subplots(figsize=(10.6, 6.2), constrained_layout=True)
    x = np.logspace(-1, 4, 600)
    ax.loglog(x, np.minimum(312, x*1.935), color='#26394a', lw=2.4,
              label='A100 PCIe 80GB nominal roof: 1,935 GB/s · 312 TOP/s')
    ax.loglog(x[x < 312/1.935], x[x < 312/1.935]*1.935*.5,
              color='#a4aeb6', ls='--', lw=1, label='50% of nominal HBM bandwidth')
    colors = {'early_decode': '#287fb8', 'long_context_decode': '#d18c26',
              'after_continuation_admission': '#54854d', None: '#777777'}
    names = {'early_decode': 'Early decode', 'long_context_decode': 'Long context decode',
             'after_continuation_admission': 'After continuation admission', None: 'Unassigned sample'}
    active = {w['phase']: int(w['before']['vllm:num_requests_running']) for w in result['windows']}
    annotations = {'early_decode': (f"Early ({active.get('early_decode', '?')} active)", (14, -8)),
                   'long_context_decode': (f"Long context ({active.get('long_context_decode', '?')})", (-110, 22)),
                   'after_continuation_admission': (f"Continuation ({active.get('after_continuation_admission', '?')})", (-122, -28)),
                   None: ('Sample', (12, 8))}
    phases = list(dict.fromkeys(p['phase'] for p in points))
    for phase in phases:
        selected = [p for p in points if p['phase'] == phase]
        ax.scatter([p['tensor_ops_per_dram_byte'] for p in selected],
                   [p['tensor_tops'] for p in selected], c=colors[phase], s=70,
                   edgecolors='white', linewidths=.8, zorder=4, label=names[phase])
        chosen = selected[len(selected)//2]
        label, offset = annotations[phase]
        ax.annotate(label, (chosen['tensor_ops_per_dram_byte'], chosen['tensor_tops']),
                    xytext=offset, textcoords='offset points', fontsize=10, color=colors[phase],
                    arrowprops={'arrowstyle': '-', 'color': colors[phase], 'lw': .7},
                    bbox={'facecolor': 'white', 'edgecolor': 'none', 'alpha': .88, 'pad': 1.5})
    ax.axvline(312/1.935, color='#abb5bb', lw=.8, ls=':')
    ax.text(312/1.935*1.08, 2, 'Ridge: 161 ops/byte', rotation=90, color='#7c8790', fontsize=9)
    ax.set(xlabel='Measured BF16 dense Tensor Core operations / measured DRAM byte',
           ylabel='Measured BF16 dense Tensor Core throughput (TOP/s)',
           title='Core v1 decode on the A100: measured roofline samples', xlim=(1, 1000), ylim=(1, 600))
    ax.grid(True, which='major', alpha=.16)
    ax.legend(loc='upper left', frameon=False, fontsize=9)
    fig.savefig(folder/'roofline.png', dpi=190)
    fig.savefig(folder/'roofline.svg')
    fig.savefig(folder/'roofline.pdf')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    short = {'early_decode': 'Early', 'long_context_decode': 'Long context',
             'after_continuation_admission': 'Continuation', None: 'Sample'}
    labels = [f"{short[p['phase']]}\n{active.get(p['phase'], '?')} active" for p in points]
    for ax, key, scale, ylabel in [(axes[0], 'dram_bytes', 1e9, 'DRAM traffic per graph (GB)'),
                                   (axes[1], 'duration_us', 1000, 'GPU graph duration (ms)')]:
        ax.bar(labels, [p[key]/scale for p in points],
               color=[colors[p['phase']] for p in points], width=.58)
        for i, p in enumerate(points):
            ax.text(i, p[key]/scale+.12, f'{p[key]/scale:.2f}', ha='center', fontsize=11)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, max(p[key]/scale for p in points)*1.22)
        ax.grid(axis='y', alpha=.15); ax.set_axisbelow(True)
    fig.suptitle('Longer contexts increase traffic as the active batch shrinks')
    fig.savefig(folder/'decode_cost.png', dpi=190)
    fig.savefig(folder/'decode_cost.svg')
    plt.close(fig)
    for name in ('roofline.svg', 'decode_cost.svg'):
        svg = folder/name
        svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('folder', type=Path)
    args = parser.parse_args()
    result = analyze(args.folder)
    plot(args.folder, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('points', 'windows')}, indent=2))


if __name__ == '__main__':
    main()
