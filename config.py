"""Shared knobs. Everything reads from here."""
import os

MODEL_ID = os.getenv("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
FAKE = os.getenv("FAKE") == "1"  # FAKE=1 → no GPU/torch, canned replies
DTYPE = os.getenv("DTYPE", "bfloat16")  # float32 on CPU
PORT = int(os.getenv("PORT", "8000"))
VECTORS_DIR = "artifacts/vectors"
MOODS_FILE = "artifacts/moods.pt"
READ_LAYER = None   # layer the detector reads (None = middle)
STEER_LAYER = None  # layer the vectors are added to (None = middle)
TRACE_FILE = "runs/trace.jsonl"

SYSTEM_PROMPT = "You are a helpful assistant."
MAX_NEW_TOKENS = 400

# step 2: which detector to try first; falls back to the next (probe → llm → neutral = no steering)
#   probe = mood read from the model's activations (needs detector/build.py) · llm = model answers with JSON
DETECTOR = os.getenv("DETECTOR", "probe")
MOODS = ["neutral", "curious", "confused", "frustrated", "rage", "sad", "rushed", "excited", "skeptical"]

# step 3: mood = SMOOTH * previous + (1 - SMOOTH) * new
SMOOTH = 0.6
INSTANT = {"rage"}  # jump up immediately, no smoothing
THRESHOLD = 0.3     # moods below this don't steer

# step 4: mood → behaviour weights. Each behaviour = one file in VECTORS_DIR (missing → skipped).
MAPPING = {
    "neutral":    {},
    "curious":    {"detailed": 1.0},
    "confused":   {"simple": 1.0},
    "frustrated": {"concise": 0.7, "careful": 0.5},
    "rage":       {"calm": 1.0, "concise": 0.5},
    "sad":        {"warm": 1.0},
    "rushed":     {"concise": 1.0},
    "excited":    {"energetic": 0.7},
    "skeptical":  {"careful": 0.6, "evidence": 1.0},
}
