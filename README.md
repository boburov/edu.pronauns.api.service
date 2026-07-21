# 🎙️ Pronunciation Assessment Service

An offline, fully open-source pronunciation scorer. Record a single word in the
browser, and the API compares what you *said* against how the word *should*
sound, returning an accuracy score from 0–100.

## How it works

```
audio (webm/ogg/wav)                target word + language
        │                                     │
   ffmpeg → 16kHz mono                  phonemizer / espeak-ng
        │                                     │
   wav2vec2 phoneme model            reference phoneme sequence
  (facebook/wav2vec2-lv-60-espeak-cv-ft)      │
        └──────────────┬──────────────────────┘
                       ▼
        token-level Levenshtein ratio → accuracy %
```

Both the predicted and reference phonemes live in the same espeak IPA space, so
they are directly comparable.

## Supported languages

English (`en-us`), Russian (`ru`), Turkish (`tr`), Spanish (`es`),
French (`fr-fr`), German (`de`), Arabic (`ar`).

Add more by extending `SUPPORTED_LANGUAGES` in `app.py` (map a code to an
espeak-ng voice).

## Install

### 1. System dependencies (required)

`espeak-ng` powers phonemization and `ffmpeg` decodes browser audio.

```bash
# Debian / Ubuntu
sudo apt update && sudo apt install -y espeak-ng ffmpeg

# macOS (Homebrew)
brew install espeak ffmpeg
```

### 2. Python dependencies

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> The first request (or startup) downloads the wav2vec2 model (~1 GB) from the
> Hugging Face Hub and caches it locally. Subsequent runs are fully offline.

## Run

```bash
uvicorn app:app --reload
```

Then open **http://127.0.0.1:8000** in a browser, pick a language, type a word,
and tap the record button.

## API

### `POST /assess`
multipart/form-data: `audio` (file), `word` (str), `language` (code)

```json
{
  "word": "hello",
  "language": "en-us",
  "accuracy": 87.5,
  "predicted_phonemes": "h ə l oʊ",
  "reference_phonemes": "h ə l oʊ",
  "verdict": "excellent"
}
```

Verdict thresholds: `≥85 excellent`, `≥70 good`, `≥50 fair`, else `poor`.

### `GET /languages`
Lists supported languages.

### `GET /health`
Reports whether the model is loaded.

## cURL example

```bash
curl -X POST http://127.0.0.1:8000/assess \
  -F "audio=@hello.wav" \
  -F "word=hello" \
  -F "language=en-us"
```

## Concurrency & scaling (CPU, bursty traffic)

The heavy pipeline (ffmpeg + wav2vec2 inference) is fully blocking, so it runs
in a worker thread — **never on the event loop**. This keeps the API responsive
(`/health`, `/languages`, static files answer in milliseconds) even while many
assessments are in flight.

Load is bounded by two knobs (env vars):

| Env var | Default | Meaning |
|---------|---------|---------|
| `MAX_CONCURRENCY` | `cores - 1` | Assessments running the heavy pipeline at once |
| `MAX_QUEUE` | `MAX_CONCURRENCY * 40` | Accepted-but-waiting requests before shedding load |
| `TORCH_THREADS` | `1` | Intra-op threads per inference (low = better under concurrency) |
| `MAX_UPLOAD_BYTES` | `5 MB` | Reject larger uploads with `413` |

When the queue is full, extra requests get a clean **`503`** ("Server is at
capacity, retry shortly") instead of piling up and exhausting memory.

**Measured on a 10-core CPU (single process):**

| Concurrent requests | Throughput | `/health` latency during burst |
|--------------------|-----------|-------------------------------|
| 1                  | —         | — |
| 50 (warm)          | ~22 req/s | median 6 ms, max 20 ms |
| 100 (warm)         | ~22 req/s | median 13 ms, max 34 ms |

Throughput is flat from ~9 concurrent onward: **the CPU is saturated at ~22 req/s**,
which is this machine's ceiling. A single word takes ~0.2 s warm.

**Handling a 1000-request burst:**

- One 10-core box drains ~22 req/s → a 1000 burst takes ~45 s. Raise `MAX_QUEUE`
  (e.g. `2000`) to accept them all and let clients wait, or keep the `503` +
  client-side retry with backoff.
- To drain a 1000 burst in a few seconds you need more throughput than one CPU
  box gives. Next levers (in order):
  1. **Horizontal scale** — run N replicas behind a load balancer (~22 req/s
     each). Nine boxes ≈ 200 req/s ≈ 1000 in ~5 s.
  2. **GPU + dynamic batching** — one GPU batching many clips per forward pass
     reaches 100–300 req/s alone.
  3. **Decouple with a queue** (Redis/RabbitMQ) + a worker pool, returning
     results via WebSocket/polling — smooths bursts against fixed capacity.
  4. **Model optimization** — ONNX Runtime / int8 quantization for faster CPU
     inference, and a distilled/smaller model.

> Note: adding more worker *processes* on the **same** box won't raise
> throughput here — the cores are already saturated. Add processes only when
> adding machines/cores.

## Error handling

All expected failures return clean 4xx JSON (`{"detail": "..."}`), never a
stack trace:

- Unsupported language → 400
- Empty / corrupt / unreadable audio → 400
- Audio longer than 10 seconds → 400
- Model still loading → 503
