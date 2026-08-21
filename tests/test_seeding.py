"""Seeding determinism -- the backbone of a reproducible SBI campaign.
Formalizes what tools/check_seeding.py checks, minus the torch/data parts, so
it runs anywhere."""

import numpy as np

from sbi.seeding import set_all_seeds


def test_set_all_seeds_makes_numpy_deterministic():
    set_all_seeds(42, deterministic=False, verbose=False)
    a = np.random.rand(5)
    set_all_seeds(42, deterministic=False, verbose=False)
    b = np.random.rand(5)
    assert np.allclose(a, b), "same seed must reproduce the same numpy draws"


def test_different_seeds_differ():
    set_all_seeds(0, deterministic=False, verbose=False)
    a = np.random.rand(5)
    set_all_seeds(1, deterministic=False, verbose=False)
    b = np.random.rand(5)
    assert not np.allclose(a, b), "different seeds should give different draws"
