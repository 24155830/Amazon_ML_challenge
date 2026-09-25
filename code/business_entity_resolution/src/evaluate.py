"""
Local re-implementation of the leaderboard metric so you can validate
before spending a submission: per-Source-1-entity F_0.5, macro-averaged.
Singletons (no true matches) score 1.0 if predicted empty, else 0.0.
"""
import pandas as pd


def _parse_ids(cell):
    if pd.isna(cell) or str(cell).strip() == "":
        return set()
    return set(x.strip() for x in str(cell).split(",") if x.strip())


def f_beta_macro(pred_df, truth_df, beta=0.5,
                  pred_id_col="source1_entity_id", pred_match_col="matched_entity_ids",
                  truth_id_col="source1_entity_id", truth_match_col="matched_entity_ids"):
    pred_map = dict(zip(pred_df[pred_id_col], pred_df[pred_match_col]))
    truth_map = dict(zip(truth_df[truth_id_col], truth_df[truth_match_col]))

    scores = []
    for s1_id, truth_cell in truth_map.items():
        truth_set = _parse_ids(truth_cell)
        pred_set = _parse_ids(pred_map.get(s1_id, ""))

        if not truth_set and not pred_set:
            scores.append(1.0)
            continue
        if not truth_set and pred_set:
            scores.append(0.0)
            continue

        tp = len(truth_set & pred_set)
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(truth_set) if truth_set else 0.0

        if precision == 0 and recall == 0:
            scores.append(0.0)
            continue

        f = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall + 1e-12)
        scores.append(f)

    return sum(scores) / len(scores) if scores else 0.0
