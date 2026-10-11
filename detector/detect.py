"""Step 2: message → mood scores 0..1.  Owner: P3.
v0 = regex + simple signals. v1: swap in a probe on model activations, keep the signature."""
import re

PATTERNS = {
    "curious":    r"\b(why|how (does|do|come)|what if|curious|interesting)\b",
    "confused":   r"(don'?t|do not) (get|understand)|what do you mean|confus|\blost\b|\bhuh\b",
    "frustrated": r"\b(no|wrong|again|still|doesn'?t work|not working|i said)\b",
    "rage":       r"\b(wtf|fuck\w*|stupid|useless|damn)\b|!!+",
    "sad":        r"give up|i'?m (so )?(bad|dumb|useless)|hopeless|\btired\b|\bsad\b",
    "rushed":     r"\b(quick(ly)?|asap|just give|fast|tl;?dr)\b",
    "excited":    r"\b(awesome|amazing|love (it|this)|let'?s go)\b|🎉|🚀",
    "skeptical":  r"are you sure|\bsource|really\?|\bprove\b|\bdoubt",
}


def detect(msg: str, history: list) -> dict[str, float]:
    low = msg.lower()
    hits = {m: len(re.findall(p, low)) for m, p in PATTERNS.items()}

    letters = [c for c in msg if c.isalpha()]
    if len(letters) > 8 and sum(c.isupper() for c in letters) / len(letters) > 0.6:
        hits["rage"] += 2  # SHOUTING
    last_user = next((h["content"] for h in reversed(history) if h["role"] == "user"), None)
    if last_user and last_user.strip().lower() == low.strip():
        hits["frustrated"] += 2  # repeated the same message

    scores = {m: min(1.0, 0.5 * n) for m, n in hits.items()}
    scores["neutral"] = 1.0 - max(scores.values())
    return scores
