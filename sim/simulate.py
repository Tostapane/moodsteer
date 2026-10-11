"""Scripted users per mood, steering ON vs OFF → runs/sim_<time>.jsonl.  Owner: P5.
v0 = fixed scripts. Next: LLM-driven users with hidden goals + a judge.
Run: python sim/simulate.py   (server must be up)"""
import json
import os
import time
import urllib.request
from pathlib import Path

SERVER = os.getenv("SERVER", "http://localhost:8000")

PERSONAS = {
    "frustrated_dev": ["my python script crashes with a KeyError, fix it",
                       "no that's wrong, I said it crashes on line 3",
                       "still not working, same error again"],
    "confused_beginner": ["what is a variable?", "I don't understand what you mean", "huh, I'm lost"],
    "rage": ["my build failed", "WHY IS THIS STILL BROKEN!!!", "this is useless wtf"],
    "sad": ["I've been stuck on this for days", "I'm so bad at this, I want to give up"],
    "rushed": ["quick: how do I reverse a list in python", "just give me the code asap"],
    "curious": ["how does garbage collection work?", "why does python use reference counting?"],
    "skeptical": ["is python slower than C?", "are you sure? source?"],
}


def post(path, body):
    req = urllib.request.Request(SERVER + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


if __name__ == "__main__":
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = Path(f"runs/sim_{ts}.jsonl")
    out.parent.mkdir(exist_ok=True)
    with out.open("w") as f:
        for name, msgs in PERSONAS.items():
            for steer in (True, False):
                sid = f"sim-{ts}-{name}-{'on' if steer else 'off'}"
                for turn, msg in enumerate(msgs):
                    r = post("/chat", {"session_id": sid, "msg": msg, "steer": steer})
                    f.write(json.dumps({"persona": name, "steer": steer, "turn": turn, "msg": msg, **r}) + "\n")
            print("done", name)
    print("wrote", out)
