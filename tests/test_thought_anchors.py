import math

import numpy as np
import pytest

from utils.heuristic.thought_anchors import chunk_spans, first_chunk, ta_importance, ta_kl


def chunks(text):
    return [text[s:e] for s, e in chunk_spans(text)]


def test_chunk_spans_tile_text_and_follow_reference_boundaries():
    text = 'Okay, the user asks X. Hmm, is that harmful?\n\nI think so! Let me plan the answer: first'
    spans = chunk_spans(text)
    assert ''.join(text[s:e] for s, e in spans) == text
    assert [c.strip() for c in chunks(text)] == [
        'Okay, the user asks X.', 'Hmm, is that harmful?', 'I think so!', 'Let me plan the answer: first']


def test_chunk_spans_decimal_point_is_not_a_boundary():
    assert [c.strip() for c in chunks('The rate is 6.17 percent. Next step here.')] == [
        'The rate is 6.17 percent.', 'Next step here.']


def test_chunk_spans_merges_short_chunks_forward_and_last_backward():
    merged = [c.strip() for c in chunks('Wait. That cannot be right. Ok.')]
    assert merged == ['Wait. That cannot be right. Ok.']


def test_chunk_spans_leading_and_repeated_whitespace():
    text = '\n\nFirst sentence here.\n\n\n\nSecond sentence here.'
    assert ''.join(chunks(text)) == text
    assert [c.strip() for c in chunks(text)] == ['First sentence here.', 'Second sentence here.']


def test_first_chunk():
    assert first_chunk(' So the user wants X. Then it continues.') == 'So the user wants X.'


def test_ta_kl_matches_hand_computation():
    p, q = ['a', 'a', 'b'], ['a', 'b', 'b', 'b']
    expected = 2 / 3 * math.log((3 / 5) / (2 / 6)) + 1 / 3 * math.log((2 / 5) / (4 / 6))
    assert ta_kl(p, q) == pytest.approx(expected)


def test_ta_kl_edge_cases():
    assert ta_kl(['a'] * 5, ['a'] * 7) == pytest.approx(0.0)
    assert ta_kl([], ['a']) == 0.0
    assert ta_kl(['', ''], ['a']) == 0.0
    # Clipped at 0 although the unsmoothed-weight estimator can go negative
    assert ta_kl(['a'], ['a', 'b']) >= 0.0


def test_ta_importance_pools_similar_resamples_with_next_chunk():
    answers = {0: ['r', 'r', 'c', 'c'], 1: ['c', 'c'], 2: ['c']}
    similarity = {0: [0.1, 0.2, 0.9, np.nan], 1: [0.9, 0.9], 2: [0.5]}
    out = ta_importance(answers, similarity).set_index('chunk_idx')
    assert out.loc[0, 'resampling_importance'] == pytest.approx(ta_kl(answers[0], answers[1]))
    # Dissimilar: two 'r'; kept side: chunk-1 rollouts plus the one similar resample ('c')
    assert out.loc[0, 'counterfactual_importance'] == pytest.approx(ta_kl(['r', 'r'], ['c', 'c', 'c']))
    assert out.loc[0, 'n_dissimilar'] == 2
    assert out.loc[1, 'counterfactual_importance'] == 0.0  # no dissimilar resamples
    assert out.loc[2, 'resampling_importance'] == 0.0  # last chunk
