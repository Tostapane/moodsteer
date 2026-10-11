"""End-to-end in FAKE mode, stdlib only. Run before every push: python tests/smoke.py"""
import os
import sys

os.environ["FAKE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

config.TRACE_FILE = "runs/smoke.jsonl"

from core.chat import chat  # noqa: E402
from detector.detect import detect  # noqa: E402

assert detect("WHY IS THIS STILL BROKEN!!!", [])["rage"] >= 1.0
assert detect("hello there", [])["neutral"] == 1.0

r = chat("smoke-a", "my code crashes")
assert set(r) == {"reply", "scores", "mood", "alphas"}
r = chat("smoke-a", "no that's wrong, still not working again")
assert r["mood"]["frustrated"] > 0.3 and "concise" in r["alphas"], r

r = chat("smoke-b", "THIS IS USELESS WTF!!!")
assert r["mood"]["rage"] == 1.0 and r["alphas"]["calm"] > 0, r  # rage skips smoothing

r = chat("smoke-c", "THIS IS USELESS WTF!!!", steer=False)
assert r["alphas"] == {}, r

print("smoke ok")
