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

# step 2: the model reads the conversation and names ONE dominant mood + intensity 0-10 (→ 0..1).
#   "probe" = experimental activation reader (needs detector/build.py), falls back to "llm".
DETECTOR = os.getenv("DETECTOR", "llm")
MOODS = ["neutral", "curious", "confused", "frustrated", "rage", "sad", "rushed", "excited", "skeptical"]

# step 3: mood = SMOOTH * previous + (1 - SMOOTH) * new   (low = follows the current message closely)
SMOOTH = 0.3
THRESHOLD = 0.3  # moods below this don't steer

# step 4: mood → axis weights. Each axis is one vector (vectors/build.py AXES) with two poles:
#   + pushes toward the first pole, − toward the second. Missing vectors are skipped.
MAPPING = {
    "neutral":    {},
    "curious":    {"cautious": 0.5},
    "confused":   {"simple": 1.0, "structured": 0.7},
    "frustrated": {"concise": 0.8, "direct": 0.6, "accountable": 0.6},
    "rage":       {"calm": 1.0, "concise": 0.7, "accountable": 0.8},
    "sad":        {"warm": 1.0, "calm": 0.5},
    "rushed":     {"concise": 1.0, "direct": 1.0},
    "excited":    {"calm": -0.5},
    "skeptical":  {"cautious": 1.0},
}

# step 5: how hard each axis pushes at α = ±1 (multiplier on the vector). Pick values with: python -m vectors.tune
STRENGTH = {
    "concise": 4.0, "warm": 4.0, "calm": 4.0, "simple": 4.0,
    "direct": 4.0, "cautious": 4.0, "structured": 4.0, "accountable": 4.0,
}
