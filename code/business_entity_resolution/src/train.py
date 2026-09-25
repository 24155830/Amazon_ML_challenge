"""
End-to-end training:
  1. Load train_source{1,2,3}.tsv + train_ground_truth.tsv
  2. Normalize text
  3. Run blocking to get candidate pairs (this doubles as a recall-ceiling
     check: print what fraction of true matches survive blocking)
  4. Label candidates against ground truth (positive / negative)
  5. Group-split by source1_entity_id (never split a S1 entity's pairs
     across train/val -- that would leak)
  6. Train a LightGBM binary classifier
  7. Sweep the decision threshold on the validation split to maximize the
     macro-averaged F_0.5 (the actual leaderboard metric), not accuracy/AUC
  8. Save model + threshold + everything predict.py needs

Run from the code/business_entity_resolution/ directory:
    python3 src/train.py --data-dir ../../dataset --out-dir ../../artifacts
"""
import argparse
import json
import os

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from preprocess import add_normalized_columns
from blocking import generate_candidates
from features import build_feature_frame, FEATURE_COLUMNS
from evaluate import f_beta_macro


def load_sources(data_dir, split):
    s1 = pd.read_csv(os.path.join(data_dir, split, f"{split}_source1.tsv"), sep="\t")
    s2 = pd.read_csv(os.path.join(data_dir, split, f"{split}_source2.tsv"), sep="\t")
    s3 = pd.read_csv(os.path.join(data_dir, split, f"{split}_source3.tsv"), sep="\t")
    s1 = add_normalized_columns(s1)
    s2 = add_normalized_columns(s2)
    s3 = add_normalized_columns(s3)
    return s1, s2, s3


def build_lookup(df):
    return {row.entity_id: (row.name_norm, row.addr_norm, row.country_norm)
            for row in df.itertuples()}


def label_pairs(pairs_df, ground_truth_df):
    truth_sets = {}
    for row in ground_truth_df.itertuples():
        ids = set() if pd.isna(row.matched_entity_ids) or not str(row.matched_entity_ids).strip() \
            else set(x.strip() for x in str(row.matched_entity_ids).split(","))
        truth_sets[row.source1_entity_id] = ids

    labels = []
    for s1_id, cand_id in zip(pairs_df["source1_entity_id"], pairs_df["candidate_entity_id"]):
        labels.append(int(cand_id in truth_sets.get(s1_id, set())))
    return np.array(labels)


def recall_ceiling(pairs_df, ground_truth_df):
    """What fraction of ground-truth matches even made it into candidates?
    This is the hard upper bound on achievable recall -- check it BEFORE
    worrying about the classifier."""
    truth_pairs = set()
    for row in ground_truth_df.itertuples():
        if pd.isna(row.matched_entity_ids) or not str(row.matched_entity_ids).strip():
            continue
        for cid in str(row.matched_entity_ids).split(","):
            truth_pairs.add((row.source1_entity_id, cid.strip()))
    if not truth_pairs:
        return 1.0
    found = set(zip(pairs_df["source1_entity_id"], pairs_df["candidate_entity_id"]))
    hit = sum(1 for p in truth_pairs if p in found)
    return hit / len(truth_pairs)


def tune_threshold(val_s1_ids, val_pairs_df, val_probs, ground_truth_df, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.05, 0.96, 0.025)

    val_pairs_df = val_pairs_df.copy()
    val_pairs_df["prob"] = val_probs

    best_t, best_f = 0.5, -1.0
    for t in thresholds:
        kept = val_pairs_df[val_pairs_df["prob"] >= t]
        grouped = kept.groupby("source1_entity_id")["candidate_entity_id"].apply(
            lambda ids: ",".join(dict.fromkeys(ids)))
        pred_rows = [{"source1_entity_id": s1, "matched_entity_ids": grouped.get(s1, "")}
                     for s1 in val_s1_ids]
        pred_df = pd.DataFrame(pred_rows)
        gt_subset = ground_truth_df[ground_truth_df["source1_entity_id"].isin(val_s1_ids)]
        score = f_beta_macro(pred_df, gt_subset)
        if score > best_f:
            best_f, best_t = score, t
    return best_t, best_f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../../dataset")
    ap.add_argument("--out-dir", default="../../artifacts")
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading + normalizing training data...")
    s1, s2, s3 = load_sources(args.data_dir, "train")
    ground_truth = pd.read_csv(os.path.join(args.data_dir, "train", "train_ground_truth.tsv"),
                                sep="\t")

    print("Running blocking on train set...")
    pairs_df = generate_candidates(s1, s2, s3)
    print(f"  {len(pairs_df)} candidate pairs generated for {s1['entity_id'].nunique()} "
          f"S1 entities")
    print(f"  recall ceiling from blocking: {recall_ceiling(pairs_df, ground_truth):.4f}")
    print("  (this is the max recall possible after blocking -- if it's low, raise "
          "top_k / lower min_sim in blocking.py before touching the model)")

    print("Labeling candidate pairs against ground truth...")
    labels = label_pairs(pairs_df, ground_truth)
    print(f"  positives: {labels.sum()} / {len(labels)} "
          f"({labels.mean() * 100:.2f}% positive rate)")

    print("Building features...")
    s1_lookup = build_lookup(s1)
    other_lookup = {**build_lookup(s2), **build_lookup(s3)}
    feat_df = build_feature_frame(pairs_df, s1_lookup, other_lookup)

    print("Group-splitting by source1_entity_id...")
    gss = GroupShuffleSplit(n_splits=1, test_size=args.val_fraction, random_state=args.seed)
    train_idx, val_idx = next(gss.split(feat_df, labels, groups=pairs_df["source1_entity_id"]))

    X_train, y_train = feat_df.iloc[train_idx], labels[train_idx]
    X_val, y_val = feat_df.iloc[val_idx], labels[val_idx]
    val_pairs = pairs_df.iloc[val_idx].reset_index(drop=True)

    val_s1_ids = pairs_df.iloc[val_idx]["source1_entity_id"].unique().tolist()
    # include val-split S1 entities that got ZERO candidates too, so the
    # threshold sweep is scored against the true singleton rate
    all_val_s1 = set(s1["entity_id"]) & (set(val_s1_ids) | (
        set(s1["entity_id"]) - set(pairs_df["source1_entity_id"])))
    val_s1_ids = sorted(all_val_s1) if all_val_s1 else val_s1_ids

    print("Training LightGBM classifier...")
    pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    model = lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=pos_weight,
        random_state=args.seed,
    )
    model.fit(X_train[FEATURE_COLUMNS], y_train,
              eval_set=[(X_val[FEATURE_COLUMNS], y_val)],
              eval_metric="auc",
              callbacks=[lgb.early_stopping(30, verbose=False)])

    print("Scoring validation candidates + tuning threshold for F_0.5...")
    val_probs = model.predict_proba(X_val[FEATURE_COLUMNS])[:, 1]
    best_t, best_f = tune_threshold(val_s1_ids, val_pairs, val_probs, ground_truth)
    print(f"  best threshold: {best_t:.3f}  ->  validation F_0.5 (macro): {best_f:.4f}")

    print("Saving artifacts...")
    joblib.dump(model, os.path.join(args.out_dir, "model.joblib"))
    with open(os.path.join(args.out_dir, "config.json"), "w") as f:
        json.dump({"threshold": float(best_t), "feature_columns": FEATURE_COLUMNS}, f, indent=2)
    print(f"Done. Model + config written to {args.out_dir}/")


if __name__ == "__main__":
    main()
