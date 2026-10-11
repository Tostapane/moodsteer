"""Steps 1-7 for one turn.  Owner: P1."""
import json
import threading
import time
from pathlib import Path

import config
from core.controller import smooth, to_alphas
from core.model import get_model
from detector.detect import detect

SESSIONS = {}            # session_id → {"history": [...], "mood": {...}}
LOCK = threading.Lock()  # one turn at a time: steering is global state on the model


def chat(session_id: str, msg: str, steer: bool = True) -> dict:
    with LOCK:
        t = time.time()
        s = SESSIONS.setdefault(session_id, {"history": [], "mood": {}})
        scores = detect(msg, s["history"])                     # 2
        s["mood"] = smooth(s["mood"], scores)                  # 3
        alphas = to_alphas(s["mood"]) if steer else {}         # 4
        messages = [{"role": "system", "content": config.SYSTEM_PROMPT}, *s["history"],
                    {"role": "user", "content": msg}]
        model = get_model()
        model.set_alphas(alphas)                               # 5
        reply = model.generate(messages)                       # 6
        s["history"] += [{"role": "user", "content": msg}, {"role": "assistant", "content": reply}]
        out = {"reply": reply, "scores": scores, "mood": s["mood"], "alphas": alphas}
        log({"session_id": session_id, "steer": steer, "msg": msg, "latency": time.time() - t, **out})  # 7
        return out


def reset(session_id: str):
    SESSIONS.pop(session_id, None)


def log(row):
    path = Path(config.TRACE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row) + "\n")
