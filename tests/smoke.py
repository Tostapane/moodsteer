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
            last = messages[-1]["content"].split("USER (last message):")[-1]
            return self.answer or ('{"mood": "rage", "intensity": 10}' if "WTF" in last else
                                   '{"mood": "frustrated", "intensity": 9}' if "wrong" in last else
                                   '{"mood": "neutral", "intensity": 0}')
        return super().generate(messages, max_new_tokens, on_token)


core.model._model = JsonModel()

r = chat("smoke-b", "my code crashes")
assert set(r) == {"reply", "scores", "alphas"} and r["alphas"] == {}, r
r = chat("smoke-b", "no that's wrong")  # 1st mood message: 100% → concise = .9 × .8
assert r["scores"]["frustrated"] == 0.9 and r["scores"]["rage"] == 0.0, r  # one dominant mood
assert abs(r["alphas"]["concise"] - 0.72) < 1e-9, r
r = chat("smoke-b", "ok, next topic")  # neutral → memory fades slowly, steering stays on
assert abs(r["alphas"]["concise"] - 0.648) < 1e-9, r
r = chat("smoke-b", "still wrong")  # 2nd mood message: 50/50
assert abs(r["alphas"]["concise"] - (0.5 * 0.648 + 0.5 * 0.72)) < 1e-9, r
r = chat("smoke-b", "wrong again")  # 3rd: capped at 40% new
assert abs(r["alphas"]["concise"] - (0.6 * 0.684 + 0.4 * 0.72)) < 1e-9, r

r = chat("smoke-c", "WTF THIS IS USELESS")
assert abs(r["alphas"]["calm"] - 1.0) < 1e-9, r

r = chat("smoke-d", "WTF", steer=False)
assert r["alphas"] == {}, r

events = []
r = chat("smoke-e", "WTF!!!", compare=True, on_event=events.append)
kinds = [e["type"] for e in events]
assert "baseline" in r and kinds.count("detected") == 1, kinds  # one detection for both replies
assert any(e["type"] == "token" and e["stream"] == "baseline" for e in events), events
assert "[fake | steering: none]" in r["baseline"] and "calm" in r["reply"], r

JsonModel.answer = '{"mood": "excited", "intensity": 10}'  # excited → calm −0.5: the other pole
r = chat("smoke-f", "yay")
assert r["alphas"]["calm"] < 0, r

JsonModel.answer = 'Sure! {"mood": "sad", "intensity": "x"}'  # not a number → detector gives up → neutral
assert detect("I give up", [])["neutral"] == 1.0
JsonModel.answer = '{"mood": "confused", "intensity": 30}'  # clamped to 10
s = detect("I don't get it", [])
assert s["confused"] == 1.0 and s["neutral"] == 0.0 and s["sad"] == 0.0, s
JsonModel.answer = '{"mood": "bored", "intensity": 8}'  # unknown mood → neutral
assert detect("meh", [])["neutral"] == 1.0

# CLI word wrap: no line longer than the width, words kept intact
import cli  # noqa: E402

turn = cli.Turn(None, None, 0)
turn.col = cli.INDENT
text = "".join(turn.wrap(chunk) for chunk in ["Waterfalls are natural ", "formations where a river ",
                                               "flows over a steep drop. " * 12, "\nNext line."])
lines = text.split("\n")
assert all(len(line) <= cli.MAX_WIDTH for line in lines) and len(lines) > 3, lines
assert "steep drop." in text and lines[-1].strip() == "Next line.", lines

print("smoke ok")
