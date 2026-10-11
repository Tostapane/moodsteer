"""Build steering vectors: mean activation difference, positive persona vs negative persona (CAA).  Owner: P2.
Run on the GPU box: python -m vectors.build [behaviour ...]   → artifacts/vectors/<behaviour>.pt
Restart the server afterwards to load new vectors."""
import sys
from pathlib import Path

import torch

import config
from core.model import get_model

# behaviour → (positive persona, negative persona). Names must match config.MAPPING.
PERSONAS = {
    "concise":   ("You answer in as few words as possible.", "You answer at great length with lots of detail."),
    "detailed":  ("You answer thoroughly, with detail and examples.", "You answer in as few words as possible."),
    "simple":    ("You explain to a beginner, step by step, in plain words.", "You explain with dense expert jargon."),
    "careful":   ("You are careful and double-check every claim.", "You answer hastily without checking anything."),
    "calm":      ("You are calm, patient and de-escalating.", "You are agitated and defensive."),
    "warm":      ("You are warm, kind and encouraging.", "You are cold and indifferent."),
    "energetic": ("You are enthusiastic and energetic!", "You are flat and monotone."),
    "evidence":  ("You back every claim with evidence and sources.", "You make claims with no justification."),
}
QUESTIONS = [
    "How do I sort a list in Python?", "Why is the sky blue?", "My code throws a KeyError, what do I do?",
    "Explain what an API is.", "How do I center a div?", "What's the difference between TCP and UDP?",
    "How can I learn to cook?", "Why does my program run slowly?",
]
MAX_ALPHA = 4.0  # tune per vector: raise until output changes, lower if it turns to garbage


def build(name, m):
    pos, neg = PERSONAS[name]

    def fmt(persona, q):
        return m.tok.apply_chat_template([{"role": "system", "content": persona}, {"role": "user", "content": q}],
                                         tokenize=False, add_generation_prompt=True)

    layer = len(m.layers) // 2
    diff = m.hidden([fmt(pos, q) for q in QUESTIONS], layer) - m.hidden([fmt(neg, q) for q in QUESTIONS], layer)
    return {"vec": diff.mean(0), "layer": layer, "max_alpha": MAX_ALPHA}


if __name__ == "__main__":
    m = get_model()
    Path(config.VECTORS_DIR).mkdir(parents=True, exist_ok=True)
    for name in sys.argv[1:] or PERSONAS:
        torch.save(build(name, m), Path(config.VECTORS_DIR) / f"{name}.pt")
        print("saved", name)
