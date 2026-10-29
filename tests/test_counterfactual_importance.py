import math

import pytest

from utils.heuristic.counterfactual_importance import (
    first_sentence,
    kl_binary,
    original_sentences,
)


def test_original_sentences_are_prefix_differences():
    prefixes = {0: 'P', 1: 'PA.', 2: 'PA. B?', 3: 'PA. B? C.'}
    assert original_sentences(prefixes) == {0: 'A.', 1: ' B?', 2: ' C.'}


def test_original_sentences_rejects_non_nested_prefixes():
    with pytest.raises(AssertionError):
        original_sentences({0: 'PA', 1: 'PB.'})


def test_first_sentence():
    assert first_sentence('Okay, so the user wants X. Then Y.') == 'Okay, so the user wants X.'
    assert first_sentence('no terminator') == 'no terminator'


def test_kl_binary_zero_for_identical_counts():
    assert kl_binary(3, 10, 3, 10) == pytest.approx(0.0)


def test_kl_binary_matches_laplace_formula():
    p = [(7 + 1) / 12, (3 + 1) / 12]
    q = [(1 + 1) / 12, (9 + 1) / 12]
    expected = sum(pi * math.log(pi / qi) for pi, qi in zip(p, q))
    assert kl_binary(7, 10, 1, 10) == pytest.approx(expected)


def test_kl_binary_is_asymmetric_with_unequal_n():
    assert kl_binary(2, 3, 1, 10) != pytest.approx(kl_binary(1, 10, 2, 3))


def test_perm_pvalue_small_for_clear_flip_and_large_for_no_effect():
    import numpy as np

    from utils.heuristic.counterfactual_importance import perm_pvalue
    rng = np.random.default_rng(0)
    flip = perm_pvalue(np.ones(20, bool), np.zeros(20, bool), 999, rng)
    same = perm_pvalue(np.r_[np.ones(10), np.zeros(10)].astype(bool), np.r_[np.ones(10), np.zeros(10)].astype(bool),
                       999, rng)
    assert flip < 0.01
    assert same > 0.5


def test_equivalence_helpers():
    from utils.heuristic.sentence_equivalence import parse_label, reasoning_context
    assert parse_label('Final answer: DIFFERENT') == 'DIFFERENT'
    assert parse_label('same') == 'SAME'
    assert parse_label('maybe') is None
    assert reasoning_context('<｜User｜>q<｜Assistant｜><think>\nOkay. So.') == 'Okay. So.'
    assert reasoning_context('<|channel|>analysis<|message|>We need') == 'We need'
    assert reasoning_context('no marker') == ''
