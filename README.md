# Moodsteer

Detect the user's mood → steer the model's activations → better next reply.

1. user sends message → 2. detect mood → 3. smooth over turns → 4. map mood → behaviours
→ 5. steer (h += Σ α·v) → 6. generate → 7. log + show

## Run

```bash
# no GPU: fake model, stdlib only
FAKE=1 python -m core.server
python tests/smoke.py            # run before every push

# GPU box (python 3.11/3.12)
python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt
python -m vectors.build          # → artifacts/vectors/*.pt
python -m core.server            # MODEL_ID=... to change model

# clients (laptop)
pip install gradio && SERVER=http://<gpu-box>:8000 python ui/app.py
SERVER=http://<gpu-box>:8000 python sim/simulate.py
```

## Who owns what

| | folder | does |
|---|---|---|
| P1 | `core/`, `config.py` | server, steering hook, smoothing, mapping |
| P2 | `vectors/` | steering vectors + α tuning |
| P3 | `detector/` | `detect(msg, history) → {mood: 0..1}` |
| P4 | `ui/` | side-by-side demo + pitch |
| P5 | `sim/` | data generation, simulated users, scoring |

Rules: push only to your folder · interface changes → tell P1 · missing piece = no steering, never a crash.
# moodsteer
