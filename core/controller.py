"""Steps 3-4: smooth mood over turns, map mood → axis alphas (-1..1, sign = pole).  Owner: P1."""
import config


def smooth(prev: dict, scores: dict) -> dict:
    return {m: config.SMOOTH * prev.get(m, 0.0) + (1 - config.SMOOTH) * scores.get(m, 0.0) for m in config.MOODS}


def to_alphas(mood: dict) -> dict:
    alphas = {}
    for m, v in mood.items():
        if v < config.THRESHOLD:
            continue
        for axis, w in config.MAPPING.get(m, {}).items():
            alphas[axis] = max(-1.0, min(1.0, alphas.get(axis, 0.0) + v * w))
    return alphas
