"""Lightweight utilities for VLM image scoring."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable, Set

import numpy as np
import spacy

_NLP = spacy.load("en_core_web_sm")
_IDF = defaultdict(lambda: 1.0)


def noun_set(text: str) -> Set[str]:
    doc = _NLP(text.lower())
    return {tok.text for tok in doc if tok.pos_ in {"NOUN", "PROPN"}}


def vector(token: str) -> np.ndarray:
    return _NLP(token).vector


def idf(token: str) -> float:
    return float(_IDF[token])


def soft_jaccard(
    text_ents: Set[str],
    img_ents: Set[str],
    *,
    idf_fn: Callable[[str], float] = idf,
    thresh: float = 0.5,
) -> float:
    def best_sim(e: str, others: Iterable[str]) -> float:
        ve = vector(e)
        if not ve.any():
            return 0.0
        ve_norm = np.linalg.norm(ve) + 1e-8
        sims = []
        for o in others:
            vo = vector(o)
            if not vo.any():
                continue
            sim = float(ve @ vo / (ve_norm * (np.linalg.norm(vo) + 1e-8)))
            sims.append(sim)
        best = max(sims, default=0.0)
        return best if best >= thresh else 0.0

    inter_score = 0.0
    for e in text_ents:
        inter_score += idf_fn(e) * best_sim(e, img_ents)
    for e in img_ents:
        inter_score += idf_fn(e) * best_sim(e, text_ents)

    union_weight = sum(idf_fn(e) for e in (text_ents | img_ents))
    return 0.0 if union_weight == 0 else inter_score / union_weight
