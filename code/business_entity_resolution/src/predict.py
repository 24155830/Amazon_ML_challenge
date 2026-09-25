"""
Inference on the test set. Produces BOTH required output files:
  output/candidate_pairs.tsv   (the blocking stage's candidate set)
  output/matching_results.tsv  (final thresholded predictions)

Run from the code/business_entity_resolution/ directory:
    python3 src/predict.py --data-dir ../../dataset --artifacts-dir ../../artifacts \
        --out-dir ../../output
"""
import argparse
import json
import os

import joblib
import pandas as pd

from preprocess import add_normalized_columns
from blocking import generate_candidates, candidates_to_tsv_rows
from features import build_feature_frame


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="../../dataset")
    ap.add_argument("--artifacts-dir", default="../../artifacts")
    ap.add_argument("--out-dir", default="../../output")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading + normalizing test data...")
    s1, s2, s3 = load_sources(args.data_dir, "test")

    print("Running blocking on test set...")
    pairs_df = generate_candidates(s1, s2, s3)

    print("Writing candidate_pairs.tsv...")
    cand_rows = candidates_to_tsv_rows(s1["entity_id"].tolist(), pairs_df)
    cand_rows.to_csv(os.path.join(args.out_dir, "candidate_pairs.tsv"), sep="\t", index=False)

    print("Loading trained model...")
    model = joblib.load(os.path.join(args.artifacts_dir, "model.joblib"))
    with open(os.path.join(args.artifacts_dir, "config.json")) as f:
        config = json.load(f)
    threshold = config["threshold"]
    feature_columns = config["feature_columns"]

    print("Scoring candidates...")
    s1_lookup = build_lookup(s1)
    other_lookup = {**build_lookup(s2), **build_lookup(s3)}
    feat_df = build_feature_frame(pairs_df, s1_lookup, other_lookup)
    pairs_df = pairs_df.copy()
    pairs_df["prob"] = model.predict_proba(feat_df[feature_columns])[:, 1]

    print(f"Applying threshold {threshold:.3f} and writing matching_results.tsv...")
    kept = pairs_df[pairs_df["prob"] >= threshold]
    grouped = kept.groupby("source1_entity_id")["candidate_entity_id"].apply(
        lambda ids: ",".join(dict.fromkeys(ids)))

    result_rows = []
    for s1_id in s1["entity_id"]:
        result_rows.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": grouped.get(s1_id, "")
        })
    result_df = pd.DataFrame(result_rows)
    result_df.to_csv(os.path.join(args.out_dir, "matching_results.tsv"), sep="\t", index=False)

    n_matched = (result_df["matched_entity_ids"] != "").sum()
    print(f"Done. {n_matched}/{len(result_df)} S1 entities matched to >=1 record.")
    print(f"Now run utils/validate_submission.py against {args.out_dir}/ before submitting.")


if __name__ == "__main__":
    main()
