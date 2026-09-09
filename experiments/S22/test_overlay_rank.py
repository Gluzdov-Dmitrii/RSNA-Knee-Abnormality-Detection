"""Rank-mix arithmetic for the S22 overlay, no DICOM/torch required."""
from __future__ import annotations

import numpy as np
import pandas as pd

REPLACEMENT = 0.40
DEFAULT_RAPTOR_W = 0.60
MENISCUS_W = (0.30, 0.60, 0.10)


def rank_pct(values: np.ndarray) -> np.ndarray:
    return pd.Series(values).rank(method="average", pct=True).to_numpy(np.float64)


def mix_target(transformer, raptor, s22, bag=None, meniscus=False):
    s22_rank = rank_pct(np.asarray(s22, dtype=np.float64))
    tr_mix = (1.0 - REPLACEMENT) * np.asarray(transformer, dtype=np.float64) + REPLACEMENT * s22_rank
    if meniscus and bag is not None:
        mixed = MENISCUS_W[0] * tr_mix + MENISCUS_W[1] * raptor + MENISCUS_W[2] * rank_pct(np.asarray(bag, dtype=np.float64))
    else:
        mixed = (1.0 - DEFAULT_RAPTOR_W) * tr_mix + DEFAULT_RAPTOR_W * np.asarray(raptor, dtype=np.float64)
    ranked = rank_pct(mixed)
    assert np.isfinite(ranked).all()
    assert ranked.min() >= 0 and ranked.max() <= 1
    return ranked


def test_identity_when_s22_matches_transformer():
    rng = np.random.default_rng(2026)
    transformer = rank_pct(rng.random(32))
    raptor = rank_pct(rng.random(32))
    expected = rank_pct((1.0 - DEFAULT_RAPTOR_W) * transformer + DEFAULT_RAPTOR_W * raptor)
    got = mix_target(transformer, raptor, transformer)
    assert np.allclose(got, expected)


def test_meniscus_uses_bag():
    rng = np.random.default_rng(7)
    transformer = rank_pct(rng.random(16))
    raptor = rank_pct(rng.random(16))
    s22 = rng.random(16)
    bag = rng.random(16)
    without = mix_target(transformer, raptor, s22, meniscus=False)
    with_bag = mix_target(transformer, raptor, s22, bag=bag, meniscus=True)
    assert not np.allclose(without, with_bag)


if __name__ == "__main__":
    test_identity_when_s22_matches_transformer()
    test_meniscus_uses_bag()
    print("ok")
