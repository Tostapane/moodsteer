"""Build steering vectors with contrastive completions (CAA).  Owner: P2.
Each vector is an AXIS between two poles (concise ↔ long): +α pushes to the first pole, −α to the second.
1. Pairs: for each question, a positive and a negative ANSWER → data/pairs/<axis>.jsonl
   {"question", "positive", "negative"}. Missing files are written by the model using the personas below
   (all in one batch); edit or replace them freely. Personas are only used to WRITE these example answers.
   Pairs are plain text, model-independent: generate once (ideally with 7B on the GPU), commit, and every
   later build only measures activations — seconds.
2. Extraction: [neutral system prompt + question + answer] goes through the model, the hidden state at
   STEER_LAYER is averaged over the ANSWER tokens only. vec = mean(positive) − mean(negative).
   No persona/instruction is in the text at this point: the vector captures the model producing the behaviour.
3. Chat time: the vector is added to the residual stream by a forward hook (core/model.py). No prompt changes.
Run: python -m vectors.build [axis ...] [--regen]   → artifacts/vectors/<axis>.pt
Then pick each axis's strength with python -m vectors.tune → config.STRENGTH."""
import json
import sys
import time
from pathlib import Path

import torch

import config
from core.model import get_model

DATA = Path("data/pairs")

# axis → (persona for the + pole, persona for the − pole). Names must match config.MAPPING / config.STRENGTH.
AXES = {
    "concise":     ("Answer in as few words as possible.",
                    "Answer at great length, with lots of detail and digressions."),
    "warm":        ("Be warm, kind and caring.",
                    "Be cold, detached and indifferent."),
    "calm":        ("Be calm, quiet and soothing.",
                    "Be extremely enthusiastic and excited, with lots of exclamation marks!"),
    "simple":      ("Explain to a complete beginner, in plain everyday words.",
                    "Explain like an expert, with dense technical jargon."),
    "direct":      ("Give the answer in the very first sentence. No preamble.",
                    "Start with a long preamble and filler before slowly getting to the point."),
    "cautious":    ("Be cautious: say what you are unsure about and admit when you don't know.",
                    "Be overconfident: state everything as certain fact and never admit doubt."),
    "structured":  ("Answer as clear numbered steps or bullet points.",
                    "Answer as one flowing paragraph of prose, with no lists."),
    "accountable": ("Plainly own the mistake in your previous answer and fix it.",
                    "Be defensive: insist your previous answer was fine and justify it."),
}
QUESTIONS = [
    "How do I sort a list in Python?", "Why is the sky blue?", "My code throws a KeyError, what do I do?",
    "Explain what an API is.", "How do I center a div?", "What's the difference between TCP and UDP?",
    "How can I learn to cook?", "Why does my program run slowly?", "What is a good way to start running?",
    "How do vaccines work?", "Can you explain recursion?", "How do I write a cover letter?",
    "What does git rebase do?", "Is coffee bad for you?", "How do I fix a merge conflict?",
    "What is inflation?", "How do I make my React app faster?", "Why did my sourdough not rise?",
    "What is a neural network?", "How should I prepare for a job interview?", "How do I read a CSV in pandas?",
    "What causes the seasons?", "How do I stop procrastinating?", "What is the difference between RAM and storage?",
]
# the "accountable" axis needs the user to be complaining about a previous answer
COMPLAINTS = [
    "Your answer about sorting lists was wrong, it didn't work.", "That explanation of the sky was incorrect.",
    "I did what you said and I still get the KeyError.", "Your API explanation confused me even more.",
    "The CSS you gave me doesn't center anything.", "You mixed up TCP and UDP in your last answer.",
    "Your cooking advice made my dish worse.", "My program is still slow after your fix.",
    "The running plan you gave me hurt my knees.", "You got the vaccine facts wrong.",
    "Your recursion example crashes with a stack overflow.", "That cover letter you wrote sounds generic.",
    "Your git rebase instructions messed up my branch.", "You told me coffee is fine but my doctor disagrees.",
    "The merge conflict steps you gave deleted my changes.", "Your definition of inflation was wrong.",
    "My React app is slower after your suggestions.", "My sourdough failed again with your tips.",
    "Your neural network explanation had an error.", "The interview tips you gave didn't work at all.",
    "Your pandas code throws an error.", "You explained the seasons wrong, it's not the distance.",
    "Your procrastination tips are useless.", "You said RAM and storage are the same, that's wrong.",
]
ANSWER_TOKENS = 60  # the style shows in the first sentences; longer answers only cost generation time


def questions(name):
    return COMPLAINTS if name == "accountable" else QUESTIONS


def generate_pairs(names, m):
    """Write data/pairs/<name>.jsonl for all `names` with ONE batched generation (faster than one batch each)."""
    convos = [[{"role": "system", "content": persona}, {"role": "user", "content": q}]
              for name in names for persona in AXES[name] for q in questions(name)]
    t = time.time()
    print(f"generating {len(convos)} example answers ({', '.join(names)})…", end=" ", flush=True)
    answers = iter(m.generate_batch(convos, max_new_tokens=ANSWER_TOKENS))
    DATA.mkdir(parents=True, exist_ok=True)
    for name in names:
        qs = questions(name)
        pos, neg = [next(answers) for _ in qs], [next(answers) for _ in qs]
        rows = [{"question": q, "positive": p, "negative": n} for q, p, n in zip(qs, pos, neg) if p and n]
        (DATA / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"done in {time.time() - t:.0f}s → {DATA}/")


def load_pairs(name):
    path = DATA / f"{name}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build(name, m):
    rows = load_pairs(name)
    layer = config.STEER_LAYER if config.STEER_LAYER is not None else len(m.layers) // 2
    prefixes = [m.chat_text([{"role": "system", "content": config.SYSTEM_PROMPT},
                             {"role": "user", "content": r["question"]}]) for r in rows]
    starts = [m.n_tokens(p) for p in prefixes]
    pos = m.acts([p + r["positive"] for p, r in zip(prefixes, rows)], layer, starts)
    neg = m.acts([p + r["negative"] for p, r in zip(prefixes, rows)], layer, starts)
    vec = pos.mean(0) - neg.mean(0)
    strength = (vec.norm() / pos.norm(dim=1).mean()).item()  # vector size vs a typical hidden state
    print(f"{name:<11} {len(rows):>5}  {'█' * max(1, round(strength * 40))} {strength:.0%}"
          + ("  weak" if strength < 0.08 else ""))
    return {"vec": vec, "layer": layer}


if __name__ == "__main__":
    t = time.time()
    regen = "--regen" in sys.argv
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(AXES)
    print(f"loading {config.MODEL_ID}…", end=" ", flush=True)
    m = get_model()
    layer = config.STEER_LAYER if config.STEER_LAYER is not None else len(m.layers) // 2
    missing = [n for n in names if regen or not (DATA / f"{n}.jsonl").exists()]
    print(f"{len(m.layers)} layers · steering layer {layer} · examples: "
          + (f"generating {len(missing)}" if missing else f"cached in {DATA}/"))
    if missing:
        generate_pairs(missing, m)
    Path(config.VECTORS_DIR).mkdir(parents=True, exist_ok=True)
    print(f"\n{'vector':<11} pairs  strength")
    for name in names:
        torch.save(build(name, m), Path(config.VECTORS_DIR) / f"{name}.pt")
    print(f"\nsaved {len(names)} vectors → {config.VECTORS_DIR}/  ({time.time() - t:.0f}s)")
