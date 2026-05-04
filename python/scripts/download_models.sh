#!/usr/bin/env bash
set -euo pipefail

ROOT="${QWEN3_AUDIO_ROOT:-/home/laiye/services/audio/qwen3_audio_api}"
MODELS_DIR="${QWEN3_AUDIO_MODELS_DIR:-$ROOT/models}"

mkdir -p "$MODELS_DIR"

echo "Downloading models into: $MODELS_DIR"

python3 - "$MODELS_DIR" <<'PY'
import json
import os
import subprocess
import sys
import urllib.request

models_dir = sys.argv[1]
models = [
    'Qwen3-TTS-12Hz-0.6B-CustomVoice',
    'Qwen3-TTS-12Hz-0.6B-Base',
    'Qwen3-TTS-12Hz-1.7B-CustomVoice',
    'Qwen3-ASR-0.6B',
    'Qwen3-ASR-1.7B',
]

for model in models:
    target = os.path.join(models_dir, model)
    os.makedirs(target, exist_ok=True)
    api = f'https://huggingface.co/api/models/Qwen/{model}'
    base = f'https://huggingface.co/Qwen/{model}/resolve/main'
    with urllib.request.urlopen(api, timeout=60) as response:
        data = json.load(response)
    files = [item.get('rfilename') for item in data.get('siblings', []) if item.get('rfilename')]
    files = [name for name in files if name not in {'.gitattributes', 'README.md'}]
    for rel in files:
        out = os.path.join(target, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        print(f'fetch {model}/{rel}', flush=True)
        subprocess.run([
            'curl', '-fL', '--retry', '8', '--retry-delay', '5', '--retry-all-errors',
            '-C', '-', '-o', out, base + '/' + rel,
        ], check=True)
PY
