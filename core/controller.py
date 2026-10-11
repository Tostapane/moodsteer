"""Steps 3-4: smooth mood over turns, map mood → behaviour alphas (0..1).  Owner: P1."""
import config


def smooth(prev: dict, scores: dict) -> dict:
    out = {}
    for m in config.MOODS:
        old, new = prev.get(m, 0.0), scores.get(m, 0.0)
        if m in config.INSTANT and new > old:
            out[m] = new
        else:
            out[m] = config.SMOOTH * old + (1 - config.SMOOTH) * new
    return out


def to_alphas(mood: dict) -> dict:
    alphas = {}
    for m, v in mood.items():
        if v < config.THRESHOLD:
            continue
        for b, w in config.MAPPING.get(m, {}).items():
            alphas[b] = min(1.0, alphas.get(b, 0.0) + v * w)
    return alphas
