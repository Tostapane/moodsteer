"""End-to-end in FAKE mode, stdlib only. Run before every push: python tests/smoke.py"""
import os
import sys

os.environ["FAKE"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402

config.TRACE_FILE = "runs/smoke.jsonl"

import core.model  # noqa: E402
from core.chat import chat  # noqa: E402
from detector.detect import PROMPT, detect  # noqa: E402

# fake model doesn't answer JSON → detector can't answer → neutral, no steering
r = chat("smoke-a", "WTF THIS IS USELESS!!!")
assert r["scores"]["neutral"] == 1.0 and r["alphas"] == {}, r


class JsonModel(core.model.FakeModel):
    """Answers the mood prompt with JSON, everything else like the fake model."""
    answer = None

    def generate(self, messages, max_new_tokens=None, on_token=None):
        if messages[0]["content"] == PROMPT:
            last = messages[-1]["content"]
            return self.answer or ('{"rage": 1}' if "WTF" in last else
                                   '{"frustrated": 0.9}' if "wrong" in last else '{"curious": 0.1}')
        return super().generate(messages, max_new_tokens, on_token)


core.model._model = JsonModel()

r = chat("smoke-b", "my code crashes")
assert set(r) == {"reply", "scores", "mood", "alphas"} and r["alphas"] == {}, r
r = chat("smoke-b", "no that's wrong")  # frustrated .9 → smoothed .36 → above threshold
assert r["mood"]["frustrated"] > 0.3 and "concise" in r["alphas"], r

r = chat("smoke-c", "WTF THIS IS USELESS")
assert r["mood"]["rage"] == 1.0 and r["alphas"]["calm"] > 0, r  # rage skips smoothing

r = chat("smoke-d", "WTF", steer=False)
assert r["alphas"] == {}, r

events = []
r = chat("smoke-e", "WTF!!!", compare=True, on_event=events.append)
kinds = [e["type"] for e in events]
assert "baseline" in r and kinds.count("detected") == 1, kinds  # one detection for both replies
assert any(e["type"] == "token" and e["stream"] == "baseline" for e in events), events
assert "[fake | steering: none]" in r["baseline"] and "calm" in r["reply"], r

JsonModel.answer = 'Sure! {"sad": "x"}'  # not a number → detector gives up → neutral
assert detect("I give up", [])["neutral"] == 1.0
JsonModel.answer = '{"confused": 0.8, "rage": 3}'
s = detect("I don't get it", [])
assert s["confused"] == 0.8 and s["rage"] == 1.0 and s["neutral"] == 0.0 and s["sad"] == 0.0, s

print("smoke ok")
