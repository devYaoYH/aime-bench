#!/usr/bin/env python3
"""Launch only the owned vLLM service under Nsight; no solver changes."""
import json
import os
import sys
from pathlib import Path

config = json.loads(Path(os.environ['CALLOSUM_NCU_CONFIG']).read_text())
os.execv(config['ncu'], [config['ncu'], *config['ncu_options'],
                        config['vllm_binary'], *sys.argv[1:],
                        '--profiler-config', '{"profiler":"cuda","max_iterations":1}'])
