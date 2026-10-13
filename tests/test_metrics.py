import math

import numpy as np

from lostfound.metrics import all_metrics, average_precision, precision_at_k, recall_at_k, reciprocal_rank


def test_perfect_ranking():
    scores = np.array([0.9, 0.8, 0.1, 0.0])
    rel = np.array([1, 1, 0, 0], bool)
    m = all_metrics(scores, rel)
    assert m["P@1"] == 1.0 and m["MRR"] == 1.0 and m["mAP"] == 1.0
    assert m["R@10"] == 1.0


def test_hand_worked_ap():
    scores = np.array([0.9, 0.8, 0.7, 0.6])
    rel = np.array([0, 1, 0, 1], bool)
    assert math.isclose(average_precision(scores, rel), 0.5)
    assert math.isclose(reciprocal_rank(scores, rel), 0.5)
    assert math.isclose(precision_at_k(scores, rel, 3), 1 / 3)
    assert math.isclose(recall_at_k(scores, rel, 3), 0.5)


def test_no_relevant_is_nan():
    assert math.isnan(average_precision(np.array([1.0, 0.5]), np.array([0, 0], bool)))
