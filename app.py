"""Pronunciation Assessment Service.

Listens to a user's recorded audio of a single word, compares it against the
correct pronunciation of that word, and returns an accuracy score (0-100).

Fully open source and offline:
  * wav2vec2 phoneme model (facebook/wav2vec2-lv-60-espeak-cv-ft) predicts the
    phonemes the user actually produced.
  * phonemizer (espeak-ng backend) produces the reference phoneme sequence for
    the target word.
  * A token-level Levenshtein ratio between the two sequences becomes the score.
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import subprocess
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Optional


def _locate_espeak_library() -> None:
    """Help phonemizer find libespeak-ng on systems where it isn't auto-detected.

    On macOS (Homebrew) the library lives under /opt/homebrew or /usr/local and
    phonemizer's dlopen probe often misses it. If PHONEMIZER_ESPEAK_LIBRARY is
    already set we respect it; otherwise we probe common locations.
    """
    if os.environ.get("PHONEMIZER_ESPEAK_LIBRARY"):
        return
    candidates = [
        "/opt/homebrew/lib/libespeak-ng.dylib",   # macOS arm64 (Homebrew)
        "/usr/local/lib/libespeak-ng.dylib",       # macOS x86_64 (Homebrew)
        "/usr/lib/x86_64-linux-gnu/libespeak-ng.so.1",  # Debian/Ubuntu
        "/usr/lib/libespeak-ng.so.1",
    ]
    for path in candidates:
        if Path(path).exists():
            os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = path
            return


_locate_espeak_library()

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Normalisation + scoring live in `phonetics.py` — pure functions over IPA
# symbols, driven by articulatory features, shared by every supported language.
from phonetics import alignment_score, normalize_phonemes

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("pronunciation")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

MODEL_ID = "facebook/wav2vec2-lv-60-espeak-cv-ft"
TARGET_SR = 16_000          # wav2vec2 expects 16 kHz mono
MAX_DURATION_S = 10.0       # reject anything longer than this
MAX_UPLOAD_BYTES = int(os.environ.get("MAX_UPLOAD_BYTES", 5 * 1024 * 1024))  # 5 MB

# --- Audio conditioning ----------------------------------------------------- #
# Recorders (phone and browser alike) start the stream at the exact moment the
# user taps, so the very first consonant is often half-swallowed. wav2vec2 also
# needs a little context before the first sound. Padding both ends with silence
# measurably recovers word-initial consonants (/h/, /b/ ...).
PAD_S = float(os.environ.get("PAD_SECONDS", 0.30))
# Phone microphones record quietly; normalising the peak gives the model a
# consistent input level regardless of device or how far the user held it.
TARGET_PEAK = 0.95
# Below this peak the clip is silence/noise, not speech.
SILENCE_PEAK = 0.005

# --- Concurrency / backpressure (tuned for CPU-only, bursty traffic) --------- #
# CPU inference is the bottleneck. We run a bounded number of assessments in
# parallel (a threadpool via asyncio.to_thread) and reject the overflow with a
# clean 503 instead of letting 1000 simultaneous requests exhaust memory.
CPU_COUNT = os.cpu_count() or 4
# How many assessments may run the heavy pipeline at the same instant.
MAX_CONCURRENCY = int(os.environ.get("MAX_CONCURRENCY", max(1, CPU_COUNT - 1)))
# How many requests may be waiting for a slot before we shed load with a 503.
MAX_QUEUE = int(os.environ.get("MAX_QUEUE", MAX_CONCURRENCY * 40))
# Per-inference intra-op threads. With many parallel requests we keep this low
# to avoid thread oversubscription (all cores fighting over one request).
TORCH_THREADS = int(os.environ.get("TORCH_THREADS", 1))

# Created lazily inside the running event loop (see lifespan).
_semaphore: Optional[asyncio.Semaphore] = None
_pending: int = 0  # requests currently accepted (queued + running)

# Supported languages -> espeak-ng voice code used by phonemizer.
# Extend this dict to add more languages (key = public code, value = espeak voice).
SUPPORTED_LANGUAGES: dict[str, dict[str, str]] = {
    "en-us": {"name": "English (US)", "espeak": "en-us"},
    "ru":    {"name": "Russian",      "espeak": "ru"},
    "tr":    {"name": "Turkish",      "espeak": "tr"},
    "es":    {"name": "Spanish",      "espeak": "es"},
    "fr-fr": {"name": "French",       "espeak": "fr-fr"},
    "de":    {"name": "German",       "espeak": "de"},
    "ar":    {"name": "Arabic",       "espeak": "ar"},
}


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #

class AssessmentError(Exception):
    """Raised for expected, user-facing failures -> becomes a clean 4xx JSON."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


# --------------------------------------------------------------------------- #
# Model state (loaded once at startup)
# --------------------------------------------------------------------------- #

class ModelState:
    processor = None
    model = None
    loaded: bool = False
    error: Optional[str] = None


state = ModelState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the wav2vec2 model exactly once, when the server starts."""
    global _semaphore
    # Bound how many CPU inferences run at once, and cap intra-op threads so
    # parallel requests don't oversubscribe the cores.
    _semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    torch.set_num_threads(TORCH_THREADS)
    logger.info(
        "Concurrency limits: max_parallel=%d, max_queue=%d, torch_threads=%d (cores=%d)",
        MAX_CONCURRENCY, MAX_QUEUE, TORCH_THREADS, CPU_COUNT,
    )

    logger.info("Loading phoneme model '%s' (first run downloads ~1 GB)...", MODEL_ID)
    try:
        from transformers import AutoModelForCTC, AutoProcessor

        state.processor = AutoProcessor.from_pretrained(MODEL_ID)
        state.model = AutoModelForCTC.from_pretrained(MODEL_ID)
        state.model.eval()
        state.loaded = True
        logger.info("Model loaded successfully.")
    except Exception as exc:  # noqa: BLE001 - surface any load failure via /health
        state.error = str(exc)
        state.loaded = False
        logger.exception("Failed to load model: %s", exc)
    yield
    # nothing to clean up


app = FastAPI(title="Pronunciation Assessment API", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(AssessmentError)
async def assessment_error_handler(_: Request, exc: AssessmentError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detail": exc.message})


# --------------------------------------------------------------------------- #
# Audio handling
# --------------------------------------------------------------------------- #

def decode_audio(raw: bytes) -> np.ndarray:
    """Decode arbitrary audio bytes (webm/ogg/wav/mp3...) to 16 kHz mono float32.

    Uses ffmpeg via a subprocess pipe, which handles every container the browser
    might send (MediaRecorder typically produces webm/opus or ogg/opus).
    """
    if not raw:
        raise AssessmentError("Empty audio upload.", 400)

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-i", "pipe:0",       # read from stdin
        "-ac", "1",           # mono
        "-ar", str(TARGET_SR),  # 16 kHz
        "-f", "wav",
        "pipe:1",             # write to stdout
    ]
    try:
        proc = subprocess.run(
            cmd,
            input=raw,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=30,
        )
    except FileNotFoundError as exc:
        raise AssessmentError(
            "ffmpeg is not installed on the server. Run: apt install ffmpeg",
            500,
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise AssessmentError("Audio decoding timed out.", 400) from exc

    if proc.returncode != 0 or not proc.stdout:
        detail = proc.stderr.decode("utf-8", "ignore").strip()[:200]
        raise AssessmentError(f"Could not decode audio (corrupt or unsupported): {detail}", 400)

    try:
        data, sr = sf.read(io.BytesIO(proc.stdout), dtype="float32")
    except Exception as exc:  # noqa: BLE001
        raise AssessmentError("Decoded audio could not be read.", 400) from exc

    if data.ndim > 1:  # safety: collapse to mono if ffmpeg ever returns stereo
        data = data.mean(axis=1)

    if data.size == 0:
        raise AssessmentError("Audio contains no samples.", 400)

    duration = data.shape[0] / float(sr or TARGET_SR)
    if duration > MAX_DURATION_S:
        raise AssessmentError(
            f"Audio too long ({duration:.1f}s). Maximum is {MAX_DURATION_S:.0f}s.",
            400,
        )
    if duration < 0.1:
        raise AssessmentError("Audio too short — please record the whole word.", 400)

    return condition_audio(data.astype(np.float32))


def condition_audio(data: np.ndarray) -> np.ndarray:
    """Level-normalise and pad the clip so the model hears the whole word.

    Two cheap fixes that together buy a lot of accuracy:
      * peak normalisation — quiet phone recordings reach the level the model
        was trained on;
      * silence padding — the first phoneme is no longer clipped by the
        recorder starting late (measured: word-initial /h/, /b/ come back).
    """
    peak = float(np.max(np.abs(data))) if data.size else 0.0
    if peak < SILENCE_PEAK:
        raise AssessmentError(
            "No speech detected — check the microphone and speak louder.", 400
        )
    data = data * (TARGET_PEAK / peak)

    pad = np.zeros(int(TARGET_SR * PAD_S), dtype=np.float32)
    return np.concatenate([pad, data, pad])


# --------------------------------------------------------------------------- #
# Phoneme extraction & comparison
# --------------------------------------------------------------------------- #

def predict_phonemes(audio: np.ndarray) -> str:
    """Run the wav2vec2 model and return space-separated phonemes."""
    if not state.loaded or state.model is None or state.processor is None:
        raise AssessmentError("Model is not loaded yet. Try again shortly.", 503)

    inputs = state.processor(
        audio, sampling_rate=TARGET_SR, return_tensors="pt", padding=True
    )
    with torch.no_grad():
        logits = state.model(inputs.input_values).logits
    predicted_ids = torch.argmax(logits, dim=-1)
    decoded = state.processor.batch_decode(predicted_ids)[0]
    return normalize_phonemes(decoded)


@lru_cache(maxsize=8192)
def reference_phonemes(word: str, espeak_code: str) -> str:
    """Get the canonical phoneme sequence for `word` via phonemizer/espeak-ng.

    Reference phonemes are deterministic for a given (word, language), so we
    cache them. Under bursty load most requests target the same handful of
    practice words, making this a large win — espeak is only invoked on a miss.
    (lru_cache is thread-safe, which matters because we call this from threads.)
    """
    from phonemizer import phonemize
    from phonemizer.separator import Separator

    try:
        out = phonemize(
            word,
            language=espeak_code,
            backend="espeak",
            separator=Separator(phone=" ", word="", syllable=""),
            strip=True,
            with_stress=False,
            preserve_punctuation=False,
        )
    except RuntimeError as exc:
        # espeak-ng missing, or voice not installed
        raise AssessmentError(
            "espeak-ng backend unavailable for this language. "
            "Install it with: apt install espeak-ng",
            500,
        ) from exc

    phonemes = normalize_phonemes(out)
    if not phonemes:
        raise AssessmentError(f"Could not phonemize the word '{word}'.", 400)
    return phonemes




def verdict_for(accuracy: float) -> str:
    if accuracy >= 85:
        return "excellent"
    if accuracy >= 70:
        return "good"
    if accuracy >= 50:
        return "fair"
    return "poor"


# --------------------------------------------------------------------------- #
# Response models
# --------------------------------------------------------------------------- #

class AssessResponse(BaseModel):
    word: str
    language: str
    accuracy: float = Field(..., ge=0, le=100)
    predicted_phonemes: str
    reference_phonemes: str
    verdict: str


class LanguageInfo(BaseModel):
    code: str
    name: str


class LanguagesResponse(BaseModel):
    languages: list[LanguageInfo]


class HealthResponse(BaseModel):
    status: str
    model_id: str
    model_loaded: bool
    error: Optional[str] = None
    # Live load info — handy for monitoring / autoscaling triggers.
    pending: int = 0
    max_concurrency: int = MAX_CONCURRENCY
    max_queue: int = MAX_QUEUE


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #

def _assess_sync(
    raw: bytes, word: str, espeak_code: str, lang: str
) -> tuple[float, str, str]:
    """The heavy, fully-blocking pipeline. Runs in a worker thread, never on the
    event loop. Returns (accuracy, predicted_phonemes, reference_phonemes)."""
    samples = decode_audio(raw)               # ffmpeg subprocess (releases GIL)
    ref = reference_phonemes(word, espeak_code)  # cached; espeak on miss
    pred = predict_phonemes(samples)          # torch inference (releases GIL)
    # `lang` enables that language's accent equivalences when scoring.
    accuracy = round(alignment_score(pred, ref, lang) * 100.0, 1)
    return accuracy, pred, ref


@app.post("/assess", response_model=AssessResponse)
async def assess(
    audio: UploadFile = File(...),
    word: str = Form(...),
    language: str = Form(...),
) -> AssessResponse:
    global _pending

    lang = language.strip().lower()
    if lang not in SUPPORTED_LANGUAGES:
        raise AssessmentError(
            f"Unsupported language '{language}'. "
            f"Supported: {', '.join(SUPPORTED_LANGUAGES)}.",
            400,
        )

    word = word.strip()
    if not word:
        raise AssessmentError("Field 'word' must not be empty.", 400)

    if not state.loaded:
        raise AssessmentError("Model is not loaded yet. Try again shortly.", 503)

    # Backpressure: if the queue is already saturated, shed load immediately with
    # a clean 503 rather than accepting work we can't process (protects memory
    # and keeps latency bounded for the requests we *do* accept).
    if _pending >= MAX_QUEUE:
        raise AssessmentError(
            "Server is at capacity. Please retry in a few seconds.", 503
        )

    # Read the upload with a hard size cap (avoids buffering huge payloads).
    raw = await audio.read()
    if len(raw) > MAX_UPLOAD_BYTES:
        raise AssessmentError(
            f"Audio file too large (> {MAX_UPLOAD_BYTES // (1024 * 1024)} MB).", 413
        )

    espeak_code = SUPPORTED_LANGUAGES[lang]["espeak"]

    _pending += 1
    try:
        assert _semaphore is not None
        # Only MAX_CONCURRENCY assessments hold the semaphore at once; the rest
        # await their turn here. The heavy work runs in a thread so the event
        # loop stays free to accept new connections and serve /health, /static.
        async with _semaphore:
            accuracy, pred, ref = await asyncio.to_thread(
                _assess_sync, raw, word, espeak_code, lang
            )
    finally:
        _pending -= 1

    return AssessResponse(
        word=word,
        language=lang,
        accuracy=accuracy,
        predicted_phonemes=pred,
        reference_phonemes=ref,
        verdict=verdict_for(accuracy),
    )


@app.get("/languages", response_model=LanguagesResponse)
async def languages() -> LanguagesResponse:
    return LanguagesResponse(
        languages=[
            LanguageInfo(code=code, name=info["name"])
            for code, info in SUPPORTED_LANGUAGES.items()
        ]
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(
        status="ok" if state.loaded else "loading",
        model_id=MODEL_ID,
        model_loaded=state.loaded,
        error=state.error,
        pending=_pending,
        max_concurrency=MAX_CONCURRENCY,
        max_queue=MAX_QUEUE,
    )


# --------------------------------------------------------------------------- #
# Static demo UI
# --------------------------------------------------------------------------- #

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def index() -> FileResponse:
    return FileResponse("static/index.html")
