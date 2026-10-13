import numpy as np


def ranking(scores):
    return np.argsort(-np.asarray(scores), kind="stable")


def precision_at_k(scores, relevant, k):
    top = ranking(scores)[:k]
    return float(np.mean([relevant[i] for i in top]))


def recall_at_k(scores, relevant, k):
    n_rel = int(np.sum(relevant))
    if n_rel == 0:
        return float("nan")
    top = ranking(scores)[:k]
    return float(np.sum([relevant[i] for i in top]) / n_rel)


def reciprocal_rank(scores, relevant):
    for r, i in enumerate(ranking(scores), start=1):
        if relevant[i]:
            return 1.0 / r
    return 0.0


def average_precision(scores, relevant):
    n_rel = int(np.sum(relevant))
    if n_rel == 0:
        return float("nan")
    hits, total = 0, 0.0
    for r, i in enumerate(ranking(scores), start=1):
        if relevant[i]:
            hits += 1
            total += hits / r
    return total / n_rel


def all_metrics(scores, relevant, ks=(1, 3, 5)):
    relevant = np.asarray(relevant, dtype=bool)
    out = {f"P@{k}": precision_at_k(scores, relevant, k) for k in ks}
    out["R@10"] = recall_at_k(scores, relevant, 10)
    out["MRR"] = reciprocal_rank(scores, relevant)
    out["mAP"] = average_precision(scores, relevant)
    return out
