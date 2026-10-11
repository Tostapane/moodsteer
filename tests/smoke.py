"""End-to-end in FAKE mode, stdlib only. Run before every push: python tests/smoke.py"""
import os
import sys

os.environ["FAKE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

config.TRACE_FILE = "runs/smoke.jsonl"

from core.chat import chat  # noqa: E402
from detector.detect import detect  # noqa: E402

assert detect("WHY IS THIS STILL BROKEN!!!", [])["rage"] >= 1.0  # fake model → not JSON → regex fallback
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

# LLM detector parsing, with a stub model that returns JSON
import core.model  # noqa: E402
from detector.detect import detect_llm  # noqa: E402


class JsonModel:
    def set_alphas(self, a): pass
    def generate(self, messages, max_new_tokens=None):
        return 'Sure! {"confused": 0.8, "rage": 3, "sad": "x"}'


core.model._model = JsonModel()
assert detect_llm("I don't get it", []) is None  # "x" isn't a number → None → regex fallback
JsonModel.generate = lambda self, m, max_new_tokens=None: '{"confused": 0.8, "rage": 3}'
s = detect_llm("I don't get it", [])
assert s["confused"] == 0.8 and s["rage"] == 1.0 and s["neutral"] == 0.0 and s["sad"] == 0.0, s
print("llm detector ok")
