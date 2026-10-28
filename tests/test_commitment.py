import numpy as np
import pandas as pd

from utils.heuristic.commitment import (
    bootstrap_mean_ci,
    commitment_position,
    prompt_level,
)


def test_committed_from_prompt():
    assert commitment_position(np.array([0.0, 0.0, 0.0, 0.0]), 0.2) == 0.0


def test_commits_after_first_sentence():
    # 4 sentences; S0 uncertain, decided from S1 onwards
    assert commitment_position(np.array([0.5, 1.0, 0.9, 1.0, 1.0]), 0.2) == 0.25


def test_late_excursion_moves_commitment():
    # agrees early, then a late wobble outside the band: commitment is after the wobble
    assert commitment_position(np.array([1.0, 1.0, 0.3, 1.0, 1.0]), 0.2) == 0.75


def test_band_is_inclusive():
    assert commitment_position(np.array([0.8, 1.0]), 0.2) == 0.0


def test_prompt_level_averages_cots():
    traj = pd.DataFrame({'prompt_idx': [1, 1, 2],
                         'means': [np.array([0.5, 1.0, 1.0]), np.array([0.0, 0.0, 0.0]), np.array([0.5, 0.5, 0.0])]})
    out = prompt_level(traj, 0.2)
    assert out[1] == 0.25 and out[2] == 1.0


def test_bootstrap_ci_contains_mean():
    mean, lo, hi = bootstrap_mean_ci(np.array([0.1, 0.2, 0.3, 0.4]), n_boot=2000)
    assert lo <= mean <= hi and np.isclose(mean, 0.25)


def test_first_entry_ignores_late_excursion():
    from utils.heuristic.commitment import first_entry_position
    assert first_entry_position(np.array([0.5, 1.0, 0.3, 1.0, 1.0]), 0.2) == 0.25
