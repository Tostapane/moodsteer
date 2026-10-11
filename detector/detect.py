"""Step 2: message → mood scores 0..1.  Owner: P3.
llm   = the served model (Qwen) reads the conversation and returns JSON scores, unsteered.
regex = keywords + simple signals; also the fallback when the LLM reply isn't valid JSON."""
import json
import re

import config

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


def detect_regex(msg: str, history: list) -> dict[str, float]:
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


PROMPT = """You read a conversation between a user and an assistant and rate the mood of the user's LAST message.
Score each mood from 0 (absent) to 1 (very strong). Several moods can be present at once.
- curious: wants to learn more, asks why/how
- confused: doesn't understand the previous answer
- frustrated: annoyed that things aren't working, corrects the assistant
- rage: angry, shouting, insulting
- sad: discouraged, wants to give up, feels bad about themselves
- rushed: in a hurry, wants a short direct answer
- excited: happy, enthusiastic
- skeptical: doubts the assistant, asks for proof
Reply with JSON only, no other text, exactly these keys:
{"curious": 0, "confused": 0, "frustrated": 0, "rage": 0, "sad": 0, "rushed": 0, "excited": 0, "skeptical": 0}"""


def detect_llm(msg: str, history: list) -> dict[str, float] | None:
    from core.model import get_model  # lazy: avoids a circular import

    convo = "\n".join(f"{h['role'].upper()}: {h['content'][:500]}" for h in history[-4:])
    convo += f"\nUSER (last message): {msg[:1000]}"
    model = get_model()
    model.set_alphas({})  # classify with the unsteered model
    reply = model.generate([{"role": "system", "content": PROMPT}, {"role": "user", "content": convo}],
                           max_new_tokens=120)
    try:
        raw = json.loads(re.search(r"\{.*\}", reply, re.S).group())
        scores = {m: min(1.0, max(0.0, float(raw.get(m, 0)))) for m in PATTERNS}
    except (AttributeError, ValueError, TypeError):
        return None
    scores["neutral"] = 1.0 - max(scores.values())
    return scores


def detect(msg: str, history: list) -> dict[str, float]:
    if config.DETECTOR == "llm":
        scores = detect_llm(msg, history)
        if scores is not None:
            return scores
    return detect_regex(msg, history)
