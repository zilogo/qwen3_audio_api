# Deploy on NVIDIA DGX Spark with Docker

This document describes the standalone Docker deployment flow that was validated on an NVIDIA DGX Spark host for the Python `qwen3_audio_api` server.

It is intended for the current deployment style:

- source checkout under `/home/laiye/Projects/audio/qwen3_audio_api`
- model and runtime assets stored outside the repository
- Docker used directly, without depending on another deployment repository

## Validated environment

- Host: NVIDIA DGX Spark
- GPU reported by `nvidia-smi`: `NVIDIA GB10`, compute capability `12.1`
- CUDA base image: `nvidia/cuda:12.8.1-devel-ubuntu24.04`
- Python implementation: `python/main.py`
- Image name used in the validated run: `qwen3-audio-api-cuda:test`

## Repository and runtime layout

Repository checkout:

```bash
/home/laiye/Projects/audio/qwen3_audio_api
```

Host-side runtime directories:

```bash
/home/laiye/services/audio/qwen3_audio_api/models
/home/laiye/services/audio/qwen3_audio_api/runtime
```

Recommended responsibility split:

- repository checkout: source code and Docker build context
- `models/`: downloaded model weights
- `runtime/`: temporary outputs, test WAV files, and helper scripts

## Why the CUDA Dockerfile needs special handling on DGX Spark

The DGX Spark host reports compute capability `12.1`, but the CUDA 12.8 toolchain exposed by `nvcc` only provides `sm_120`, not `sm_121`.

Because of that, the working Docker build uses:

- `TORCH_CUDA_ARCH_LIST="12.0"`
- `FLASH_ATTN_CUDA_ARCHS="120"`

This keeps the `flash-attn` build focused on the Blackwell target available in the toolchain and avoids compiling unrelated architectures such as `80`, `90`, `100`, or `110`.

## Models used in validation

The following models were downloaded and verified:

```text
Qwen3-TTS-12Hz-0.6B-CustomVoice
Qwen3-TTS-12Hz-0.6B-Base
Qwen3-TTS-12Hz-1.7B-CustomVoice
Qwen3-ASR-0.6B
Qwen3-ASR-1.7B
```

Default host-side model path on Spark:

```bash
/home/laiye/services/audio/qwen3_audio_api/models
```

Create the host-side directories:

```bash
mkdir -p /home/laiye/services/audio/qwen3_audio_api/models
mkdir -p /home/laiye/services/audio/qwen3_audio_api/runtime
```

Download all validated models with the repository script:

```bash
cd /home/laiye/Projects/audio/qwen3_audio_api/python
./scripts/download_models.sh
```

By default, the script downloads into:

```bash
/home/laiye/services/audio/qwen3_audio_api/models
```

You can override the destination if needed:

```bash
QWEN3_AUDIO_MODELS_DIR=/data/models ./scripts/download_models.sh
```

## Build the CUDA image

Run the build from the `python/` directory:

```bash
cd /home/laiye/Projects/audio/qwen3_audio_api/python

docker build -f Dockerfile.cuda -t qwen3-audio-api-cuda:test .
```

Notes:

- the first build is slow because `flash-attn` is compiled from source
- the validated Dockerfile installs CUDA-enabled `torch` and `torchaudio`
- `flash-attn` is built with the DGX Spark-compatible architecture settings described above

## Run a 0.6B GPU deployment

This profile is a good default for functional validation and lighter-weight serving.

```bash
docker run -d --rm \
  --gpus all \
  --name qwen3-audio-api-06 \
  -p 18000:8000 \
  -v /home/laiye/services/audio/qwen3_audio_api/models:/models \
  -v /home/laiye/services/audio/qwen3_audio_api/runtime:/runtime \
  -e TTS_CUSTOMVOICE_MODEL_PATH=/models/Qwen3-TTS-12Hz-0.6B-CustomVoice \
  -e TTS_BASE_MODEL_PATH=/models/Qwen3-TTS-12Hz-0.6B-Base \
  -e ASR_MODEL_PATH=/models/Qwen3-ASR-0.6B \
  -e QWEN_TTS_DEVICE=cuda \
  -e QWEN_TTS_DTYPE=bfloat16 \
  -e QWEN_TTS_ATTN=flash_attention_2 \
  qwen3-audio-api-cuda:test
```

## Run a 1.7B GPU deployment

This profile uses the larger TTS and ASR models.

```bash
docker run -d --rm \
  --gpus all \
  --name qwen3-audio-api-17 \
  -p 18001:8000 \
  -v /home/laiye/services/audio/qwen3_audio_api/models:/models \
  -v /home/laiye/services/audio/qwen3_audio_api/runtime:/runtime \
  -e TTS_CUSTOMVOICE_MODEL_PATH=/models/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  -e TTS_BASE_MODEL_PATH=/models/Qwen3-TTS-12Hz-0.6B-Base \
  -e ASR_MODEL_PATH=/models/Qwen3-ASR-1.7B \
  -e QWEN_TTS_DEVICE=cuda \
  -e QWEN_TTS_DTYPE=bfloat16 \
  -e QWEN_TTS_ATTN=flash_attention_2 \
  qwen3-audio-api-cuda:test
```

## Optional: run with Docker Compose

The repository `python/docker-compose.yml` has been updated to match the validated Spark deployment layout.

Export the Spark host paths first:

```bash
export QWEN3_AUDIO_MODELS_DIR=/home/laiye/services/audio/qwen3_audio_api/models
export QWEN3_AUDIO_RUNTIME_DIR=/home/laiye/services/audio/qwen3_audio_api/runtime
```

Run the 0.6B profile:

```bash
cd /home/laiye/Projects/audio/qwen3_audio_api/python
docker compose --profile cuda-06 up -d --build
```

Run the 1.7B profile:

```bash
cd /home/laiye/Projects/audio/qwen3_audio_api/python
docker compose --profile cuda-17 up -d --build
```

Bring the service down:

```bash
docker compose --profile cuda-06 down
docker compose --profile cuda-17 down
```

## Health checks

```bash
curl http://127.0.0.1:18000/health
curl http://127.0.0.1:18000/v1/models
```

Expected examples:

```json
{"status":"ok"}
```

```json
{"object":"list","data":[{"id":"qwen3-tts","object":"model","owned_by":"qwen"},{"id":"qwen3-asr","object":"model","owned_by":"qwen"}]}
```

## TTS smoke test

```bash
curl -X POST http://127.0.0.1:18000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3-tts",
    "input": "Hello from Qwen3 audio API on DGX Spark.",
    "voice": "alloy",
    "language": "English",
    "response_format": "wav"
  }' \
  --output /home/laiye/services/audio/qwen3_audio_api/runtime/tts_en.wav
```

## ASR smoke test

```bash
curl -X POST http://127.0.0.1:18000/v1/audio/transcriptions \
  -F file=@/home/laiye/services/audio/qwen3_audio_api/runtime/tts_en.wav \
  -F model=qwen3-asr
```

## Voice cloning smoke test

```bash
curl -X POST http://127.0.0.1:18000/v1/audio/speech \
  -F model=qwen3-tts \
  -F 'input=This is a cloned voice test.' \
  -F audio_sample=@/home/laiye/services/audio/qwen3_audio_api/runtime/tts_en.wav \
  -F 'audio_sample_text=Hello from Qwen3 audio API on DGX Spark.' \
  -F language=English \
  -F response_format=wav \
  --output /home/laiye/services/audio/qwen3_audio_api/runtime/clone_en.wav
```

## Validated performance on Spark

The following results were measured on the validated DGX Spark setup after model warmup.

Test text used for the TTS benchmark:

```text
Hello from the Qwen3 Audio API benchmark running on DGX Spark. This sentence is used to measure text to speech and speech recognition performance.
```

RTF means `wall time / audio duration`.

- `RTF < 1.0`: faster than real time
- `RTF = 1.0`: roughly real time
- `RTF > 1.0`: slower than real time

Measured results:

| Model | Cold start to `/health` | TTS warmup | TTS avg latency | TTS avg audio | TTS avg RTF | ASR warmup | ASR avg latency | ASR sample audio | ASR avg RTF |
|-------|--------------------------|------------|-----------------|---------------|-------------|------------|-----------------|------------------|-------------|
| `0.6B` | `48.512s` | `21.688s` | `17.970s` | `11.307s` | `1.593` | `6.518s` | `0.620s` | `10.960s` | `0.057` |
| `1.7B` | `80.627s` | `20.835s` | `23.856s` | `13.067s` | `1.825` | `7.142s` | `1.253s` | `14.080s` | `0.089` |

Interpretation:

- ASR is already much faster than real time in this setup.
- TTS is functional but still slower than real time for both `0.6B` and `1.7B`.
- `1.7B` is slower than `0.6B`, especially during cold start.
- Cold start is dominated by model loading, so long-lived containers are strongly preferred for serving.

## Operational notes

- Cold start is model-load heavy. Expect tens of seconds before `/health` turns green.
- ASR is already much faster than real time in the validated setup.
- TTS is functional but slower than real time in the validated setup.
- `flash-attn` is important on this host. Do not silently fall back unless you are intentionally debugging build issues.
- If you only need TTS or only need ASR, you can omit the unused model path environment variables.

## Logs and cleanup

View logs:

```bash
docker logs -f qwen3-audio-api-06
docker logs -f qwen3-audio-api-17
```

Stop and remove the test containers:

```bash
docker rm -f qwen3-audio-api-06
docker rm -f qwen3-audio-api-17
```

## Recommended files to keep under version control

For this deployment path, the repository should at least keep:

- `python/Dockerfile.cuda`
- `python/README.md`
- `python/DEPLOY_SPARK_DOCKER.md`
- `python/scripts/download_models.sh`

Host-local model downloads, generated WAV files, and runtime helper scripts should stay outside the repository.
