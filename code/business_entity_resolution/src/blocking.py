"""
Candidate generation (blocking).

Strategy:
  1. Group records by normalized country. This is just a bucket to shrink
     the search space -- it is NOT a hard filter on the final match (a
     record with a noisy/missing country can still be found; see the
     "unblocked" fallback below). Works for any country string, including
     ones never seen in training (e.g. France in the test set), because we
     never hard-code a country list.
  2. Within each bucket, fit a character n-gram TF-IDF vectorizer over
     name_norm + " " + addr_norm for all S1/S2/S3 records in that bucket,
     and use cosine nearest-neighbors to pull the top-K most similar S2/S3
     records for every S1 record.
  3. A small "unblocked" fallback pass handles S1 records whose country
     bucket is empty/unmatched on the other side (e.g. a typo'd or missing
     country) by searching across ALL records regardless of country.
  4. Union everything, cap total candidates per S1 record, dedupe.

Output: one row per (source1_entity_id, candidate_entity_id) with the
cosine similarity that produced it (reused later as a feature).
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


def _combined_text(df):
    return (df["name_norm"] + " " + df["addr_norm"]).values


def _knn_candidates(s1_df, other_df, top_k, min_sim):
    """Return list of (s1_entity_id, other_entity_id, cosine_sim)."""
    if len(s1_df) == 0 or len(other_df) == 0:
        return []

    vectorizer = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=1)
    all_text = list(_combined_text(s1_df)) + list(_combined_text(other_df))
    tfidf = vectorizer.fit_transform(all_text)

    s1_vec = tfidf[: len(s1_df)]
    other_vec = tfidf[len(s1_df):]

    k = min(top_k, len(other_df))
    nn = NearestNeighbors(n_neighbors=k, metric="cosine")
    nn.fit(other_vec)
    distances, indices = nn.kneighbors(s1_vec)

    results = []
    s1_ids = s1_df["entity_id"].values
    other_ids = other_df["entity_id"].values
    for i in range(len(s1_df)):
        for j, dist in zip(indices[i], distances[i]):
            sim = 1.0 - dist
            if sim >= min_sim:
                results.append((s1_ids[i], other_ids[j], float(sim)))
    return results


def generate_candidates(source1_df, source2_df, source3_df, top_k=15, min_sim=0.15,
                         max_candidates_per_entity=40):
    """
    source1_df/source2_df/source3_df must already have name_norm/addr_norm/
    country_norm columns (see preprocess.add_normalized_columns).

    Returns a DataFrame: source1_entity_id, candidate_entity_id, cosine_sim
    -- already capped and deduped, ready to write as candidate_pairs.tsv
    (after grouping into comma-joined lists) and to feed into features.py.
    """
    all_pairs = []

    # --- Pass 1: per-country blocking ---
    countries = sorted(set(source1_df["country_norm"]))
    for country in countries:
        s1_c = source1_df[source1_df["country_norm"] == country]
        s2_c = source2_df[source2_df["country_norm"] == country]
        s3_c = source3_df[source3_df["country_norm"] == country]

        all_pairs += [(s1, s2, sim, "S2") for s1, s2, sim in
                      _knn_candidates(s1_c, s2_c, top_k, min_sim)]
        all_pairs += [(s1, s3, sim, "S3") for s1, s3, sim in
                      _knn_candidates(s1_c, s3_c, top_k, min_sim)]

    pairs_df = pd.DataFrame(all_pairs, columns=["source1_entity_id", "candidate_entity_id",
                                                  "cosine_sim", "src"])

    # --- Pass 2: fallback for S1 entities that got few/no candidates ---
    # (handles noisy/mismatched country labels)
    counts = pairs_df.groupby("source1_entity_id").size()
    weak_ids = set(source1_df["entity_id"]) - set(counts[counts >= 3].index)
    if weak_ids:
        s1_weak = source1_df[source1_df["entity_id"].isin(weak_ids)]
        fb2 = [(s1, s2, sim, "S2") for s1, s2, sim in
               _knn_candidates(s1_weak, source2_df, top_k, min_sim)]
        fb3 = [(s1, s3, sim, "S3") for s1, s3, sim in
               _knn_candidates(s1_weak, source3_df, top_k, min_sim)]
        fb_df = pd.DataFrame(fb2 + fb3, columns=["source1_entity_id", "candidate_entity_id",
                                                   "cosine_sim", "src"])
        pairs_df = pd.concat([pairs_df, fb_df], ignore_index=True)

    # dedupe (keep max sim if a pair appeared via both passes)
    pairs_df = (pairs_df.sort_values("cosine_sim", ascending=False)
                .drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"]))

    # cap candidates per S1 entity without groupby.apply (which can silently
    # drop the constant grouping column on some pandas versions)
    pairs_df = pairs_df.sort_values(["source1_entity_id", "cosine_sim"], ascending=[True, False])
    pairs_df["_rank"] = pairs_df.groupby("source1_entity_id").cumcount()
    pairs_df = pairs_df[pairs_df["_rank"] < max_candidates_per_entity].drop(columns="_rank")

    return pairs_df.reset_index(drop=True)


def candidates_to_tsv_rows(source1_ids, pairs_df):
    """Build the one-row-per-S1-entity structure required for candidate_pairs.tsv."""
    grouped = pairs_df.groupby("source1_entity_id")["candidate_entity_id"].apply(
        lambda ids: ",".join(dict.fromkeys(ids))  # dedupe, preserve order
    )
    rows = []
    for s1_id in source1_ids:
        rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": grouped.get(s1_id, "")
        })
    return pd.DataFrame(rows)
