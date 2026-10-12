# Moodsteer

A chat assistant that adapts to the user's mood by steering the model's activations, not its prompt.

Every turn:
1. **Detect:** the model reads the conversation and names the user's dominant mood + intensity
   (frustrated 6/10, confused 3/10, …).
2. **Remember:** a per-axis steering state is updated: 100% the new mood on the first mood message, then 50%,
   then 40% new / 60% memory. Neutral messages fade it slowly (×0.9).
3. **Steer:** each axis vector × its α × its strength is added to the hidden state at the middle layer
   while the reply is generated. Weights and prompt are untouched.
4. **Compare:** the same question is also answered without steering, side by side.

The 8 steering axes (+ pole ↔ − pole): concise ↔ long · warm ↔ cold · calm ↔ enthusiastic ·
simple ↔ technical · direct ↔ roundabout · cautious ↔ overconfident · structured ↔ freeform ·
accountable ↔ defensive.

Each vector is built with contrastive completions: the model answers the same questions in both poles,
and the vector is the difference of the mean hidden state over the answer tokens (+ minus −).

## Run (GPU, Qwen2.5-3B-Instruct)

```bash
git clone <repo> moodsteer && cd moodsteer
pip install "transformers>=4.56" accelerate      # plus torch

# 1. build the 8 vectors (--regen rewrites the example answers in data/pairs/)
MODEL_ID=Qwen/Qwen2.5-3B-Instruct DTYPE=float16 python -m vectors.build --regen

# 2. optional: see each axis at ×2/4/6/8 to pick its strength
MODEL_ID=Qwen/Qwen2.5-3B-Instruct DTYPE=float16 python -m vectors.tune
```

3. Chat, from Python (works in a notebook cell):

```python
import os
os.environ.update(MODEL_ID="Qwen/Qwen2.5-3B-Instruct", DTYPE="float16")
import config
config.STRENGTH.update(concise=3, simple=2.5, direct=4, calm=5, structured=6, accountable=6, warm=9, cautious=11)
import cli
cli.main()      # commands: /compare /steer /reset /quit
```

Set the environment variables before `import config`. They're read once at import.

## Knobs (`config.py`)

- `MAPPING`: which axes each mood pushes, and with what sign.
- `STRENGTH`: how hard each axis pushes at α = ±1.
- `KEEP`, `HOLD`, `THRESHOLD`: steering memory.
- `SYSTEM_PROMPT`, `MAX_NEW_TOKENS`.

## Code

- `detector/detect.py`: mood detection (the model answers with JSON).
- `core/controller.py`: steering memory.
- `core/model.py`: model loading, the steering hook, generation.
- `core/chat.py`: one turn.
- `vectors/build.py`, `vectors/tune.py`: build and inspect the vectors.
- `cli.py`: the chat interface.

Before pushing, run `python tests/smoke.py`. It needs no GPU or torch.
