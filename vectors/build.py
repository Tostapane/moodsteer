"""Build steering vectors with contrastive completions (CAA).  Owner: P2.
1. Pairs: for each question, a positive and a negative ANSWER → data/pairs/<behaviour>.jsonl
   {"question", "positive", "negative"}. Missing files are written by the model using the personas below
   (all in one batch); edit or replace them freely. Personas are only used to WRITE these example answers.
   Pairs are plain text, model-independent: generate once (ideally with 7B on the GPU), commit, and every
   later build only measures activations — seconds.
2. Extraction: [neutral system prompt + question + answer] goes through the model, the hidden state at
   STEER_LAYER is averaged over the ANSWER tokens only. vec = mean(positive) − mean(negative).
   No persona/instruction is in the text at this point: the vector captures the model producing the behaviour.
3. Chat time: the vector is added to the residual stream by a forward hook (core/model.py). No prompt changes.
Run on the GPU box: python -m vectors.build [behaviour ...] [--regen]   → artifacts/vectors/<behaviour>.pt
Restart the server afterwards."""
import json
import sys
from pathlib import Path

import torch

import config
from core.model import get_model

DATA = Path("data/pairs")

# behaviour → (persona for the positive answer, persona for the negative answer). Names must match config.MAPPING.
PERSONAS = {
    "concise":   ("Answer in as few words as possible.", "Answer at great length with lots of detail and digressions."),
    "detailed":  ("Answer thoroughly, with detail and examples.", "Answer in as few words as possible."),
    "simple":    ("Explain to a complete beginner, step by step, in plain words.", "Explain with dense expert jargon."),
    "careful":   ("Be careful: state assumptions, double-check, mention caveats.", "Answer hastily and overconfidently."),
    "calm":      ("Be calm, patient and reassuring.", "Be irritated, impatient and defensive."),
    "warm":      ("Be warm, kind and encouraging.", "Be cold, curt and indifferent."),
    "energetic": ("Be enthusiastic and energetic!", "Be flat, bored and monotone."),
    "evidence":  ("Back every claim with evidence, reasons and sources.", "Make claims with no justification at all."),
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
ANSWER_TOKENS = 60  # the style shows in the first sentences; longer answers only cost generation time
MAX_ALPHA = 4.0  # strength at α = 1. Qwen 0.5B: ×4 = clearly shorter and coherent, ×5 starts breaking, ×10 = garbage.
#                 Re-tune per model: raise until the output changes, lower if it turns to garbage.


def generate_pairs(names, m):
    """Write data/pairs/<name>.jsonl for all `names` with ONE batched generation (faster than one batch each)."""
    convos = [[{"role": "system", "content": persona}, {"role": "user", "content": q}]
              for name in names for persona in PERSONAS[name] for q in QUESTIONS]
    print(f"generating {len(convos)} example answers for {', '.join(names)} (one batch)…", flush=True)
    answers = iter(m.generate_batch(convos, max_new_tokens=ANSWER_TOKENS))
    DATA.mkdir(parents=True, exist_ok=True)
    for name in names:
        pos, neg = [next(answers) for _ in QUESTIONS], [next(answers) for _ in QUESTIONS]
        rows = [{"question": q, "positive": p, "negative": n} for q, p, n in zip(QUESTIONS, pos, neg) if p and n]
        (DATA / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"generated data/pairs/{name}.jsonl")


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
    print(f"{name}: {len(rows)} pairs, layer {layer}, |vec| = {vec.norm():.1f} "
          f"({vec.norm() / pos.norm(dim=1).mean():.0%} of a typical hidden state)")
    return {"vec": vec, "layer": layer, "max_alpha": MAX_ALPHA}


if __name__ == "__main__":
    regen = "--regen" in sys.argv
    names = [a for a in sys.argv[1:] if not a.startswith("--")] or list(PERSONAS)
    m = get_model()
    missing = [n for n in names if regen or not (DATA / f"{n}.jsonl").exists()]
    if missing:
        generate_pairs(missing, m)
    Path(config.VECTORS_DIR).mkdir(parents=True, exist_ok=True)
    for name in names:
        torch.save(build(name, m), Path(config.VECTORS_DIR) / f"{name}.pt")
