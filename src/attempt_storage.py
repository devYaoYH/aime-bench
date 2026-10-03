"""Attempt-owned telemetry storage; buffered mode writes after official timing ends."""

from contextlib import contextmanager
from copy import deepcopy
import io
import json
from pathlib import Path
import time

from src.common import atomic_json


class AttemptArtifacts:
    def __init__(self, root, *, buffered=False):
        self.root, self.buffered = Path(root), buffered
        self.json_files, self.jsonl_files, self.text_files = {}, {}, {}

    def write_json(self, path, value):
        path = Path(path)
        if self.buffered:
            self.json_files[path] = deepcopy(value)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(path, value)

    def has_json(self, path):
        return Path(path) in self.json_files or Path(path).exists()

    def read_json(self, path):
        path = Path(path)
        if path in self.json_files:
            return deepcopy(self.json_files[path])
        return json.loads(path.read_text())

    def questions(self):
        paths = set(self.root.glob('trace/*/question.json'))
        paths.update(p for p in self.json_files if p.name == 'question.json')
        return [self.read_json(p) for p in sorted(paths)]

    @contextmanager
    def open_jsonl(self, path, mode='a'):
        path = Path(path)
        if self.buffered:
            if mode == 'w' or path not in self.jsonl_files:
                self.jsonl_files[path] = (mode, [])
            rows = self.jsonl_files[path][1]
            # SSE rows contain only immutable scalars; avoid copying their payloads.
            flat = path.name == 'stream.jsonl'

            class Sink:
                def append(self, row):
                    rows.append(dict(row) if flat else deepcopy(row))

            yield Sink()
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open(mode) as file:
                class Sink:
                    def append(self, row):
                        file.write(json.dumps(row, ensure_ascii=False) + '\n')
                        file.flush()
                yield Sink()

    def gpu_path(self, path):
        """Let the frozen GPU sampler retain its low-frequency file output in RAM."""
        if not self.buffered:
            return path
        owner = self

        class DeferredPath:
            @contextmanager
            def open(self, mode):
                with io.StringIO() as file:
                    try:
                        yield file
                    finally:
                        owner.text_files[Path(path)] = file.getvalue()

        return DeferredPath()

    def flush(self):
        """Call after producers stop. Successful files are released, failures retained."""
        start = time.perf_counter()
        result = {'buffered': self.buffered, 'files_written': 0,
                  'jsonl_rows': 0, 'payload_characters': 0}
        for path, value in list(self.json_files.items()):
            path.parent.mkdir(parents=True, exist_ok=True)
            atomic_json(path, value)
            del self.json_files[path]
            result['files_written'] += 1
        for path, (mode, rows) in list(self.jsonl_files.items()):
            path.parent.mkdir(parents=True, exist_ok=True)
            # Rewrite atomically, including any existing prefix for append-mode logs.
            tmp = path.with_suffix(path.suffix + '.tmp')
            try:
                with tmp.open('w') as file:
                    if mode == 'a' and path.exists():
                        with path.open() as prior:
                            for line in prior:
                                file.write(line)
                    for row in rows:
                        file.write(json.dumps(row, ensure_ascii=False) + '\n')
                tmp.replace(path)
            finally:
                tmp.unlink(missing_ok=True)
            result['jsonl_rows'] += len(rows)
            result['payload_characters'] += sum(len(r.get('data', '')) for r in rows)
            del self.jsonl_files[path]
            result['files_written'] += 1
        for path, text in list(self.text_files.items()):
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + '.tmp')
            tmp.write_text(text)
            tmp.replace(path)
            del self.text_files[path]
            result['files_written'] += 1
        result['latency_s'] = time.perf_counter() - start
        return result


class DisabledGPUSampler:
    """Explicitly missing GPU observations; benchmark mode never launches NVML polls."""
    error = None

    async def start(self):
        pass

    def stop(self):
        pass

    def window(self, start, end):
        return None


def add_benchmark_args(parser):
    parser.add_argument('--benchmark', action='store_true',
                        help='Disable optional profiling/GPU polling and buffer client traces until attempt end')
    parser.add_argument('--buffer-traces', action='store_true',
                        help='Retain client JSON/JSONL telemetry in RAM; flush after official timing ends')
    parser.add_argument('--no-gpu-telemetry', action='store_true',
                        help='Disable NVML sampling; GPU observations will be null')


def apply_benchmark_args(args):
    if args.benchmark:
        args.no_overhead_profile = args.no_gpu_telemetry = args.buffer_traces = True
    return args
