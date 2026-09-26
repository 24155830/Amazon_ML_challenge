"""
Memory-efficient candidate generation -- OPTIMIZED.

Same blocking strategy as before (country-agnostic name-prefix, name-token,
and address-number indexes, unioned together), with three concrete speedups:

1. Postings lists are capped AT BUILD TIME, not just read-time. The
   previous version stored every occurrence of every token/prefix/number
   and only sliced to the first 15 when looking it up -- so a common
   4-character name prefix shared by 300,000 records still cost 300,000
   list appends and ~2.4 MB of memory for a signal only its first 15
   entries were ever going to be used from. Stopping at 15 appends per key
   is a direct, large reduction in both build time and memory.

2. Source-1 blocking keys (prefix / tokens / address numbers) are computed
   ONCE per S1 record and reused for both the Source-2 pass and the
   Source-3 pass. The previous version recomputed them from scratch on
   both passes -- i.e. did the same regex/split work on the same 2.2M rows
   twice for no reason.

3. Regex patterns are precompiled once at module load instead of being
   re-parsed (or re-fetched from re's internal cache) on every call.

The expensive fuzzy features are still calculated later by features.py.
"""

import re
import gc
from collections import defaultdict

import pandas as pd

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]")
_DIGIT_RE = re.compile(r"\d+")

# Postings per key are capped at this many entries AT BUILD TIME (see point 1
# above). Keep this equal to (or larger than) max_from_each_key semantics
# from the original version.
MAX_POSTINGS_PER_KEY = 400


# ============================================================
# BASIC HELPERS (unchanged behavior, precompiled regex)
# ============================================================

def _name_prefix(name_norm, n=4):
    if not name_norm:
        return ""
    cleaned = _NON_ALNUM_RE.sub("", name_norm)
    return cleaned[:n]


def _name_tokens(name_norm):
    if not name_norm:
        return ()
    return tuple(t for t in name_norm.split() if len(t) >= 3)


def _address_numbers(addr_norm):
    if not addr_norm:
        return ()
    return tuple(set(_DIGIT_RE.findall(addr_norm)))


# ============================================================
# PRECOMPUTE SOURCE-1 KEYS ONCE (speedup #2)
# ============================================================

def _precompute_keys(df):
    """entity_id -> (prefix, tokens, numbers), computed once for the whole df."""
    keys = {}
    for row in df.itertuples():
        keys[row.entity_id] = (
            _name_prefix(row.name_norm),
            _name_tokens(row.name_norm),
            _address_numbers(row.addr_norm),
        )
    return keys


# ============================================================
# BUILD INDEXES (speedup #1: cap postings at build time)
# ============================================================

def _build_indexes(other_df, max_postings=MAX_POSTINGS_PER_KEY):
    prefix_index = defaultdict(list)
    token_index = defaultdict(list)
    number_index = defaultdict(list)

    # Small "already full" sets let us skip a key entirely once it has
    # reached the cap, instead of appending-then-slicing later.
    prefix_full = set()
    token_full = set()
    number_full = set()

    for row in other_df.itertuples():
        eid = row.entity_id

        prefix = _name_prefix(row.name_norm)
        if prefix and prefix not in prefix_full:
            lst = prefix_index[prefix]
            lst.append(eid)
            if len(lst) >= max_postings:
                prefix_full.add(prefix)

        for tok in _name_tokens(row.name_norm):
            if tok in token_full:
                continue
            lst = token_index[tok]
            lst.append(eid)
            if len(lst) >= max_postings:
                token_full.add(tok)

        for num in _address_numbers(row.addr_norm):
            if num in number_full:
                continue
            lst = number_index[num]
            lst.append(eid)
            if len(lst) >= max_postings:
                number_full.add(num)

    return prefix_index, token_index, number_index


# ============================================================
# CANDIDATES FOR ONE RECORD (keys passed in precomputed)
# ============================================================

def _get_candidates(prefix, tokens, numbers, prefix_index, token_index, number_index):
    candidates = set()

    if prefix:
        ids = prefix_index.get(prefix)
        if ids:
            candidates.update(ids)

    for tok in tokens:
        ids = token_index.get(tok)
        if ids:
            candidates.update(ids)

    for num in numbers:
        ids = number_index.get(num)
        if ids:
            candidates.update(ids)

    return candidates


# ============================================================
# GENERATE CANDIDATES BETWEEN TWO SOURCES
# ============================================================

def _generate_source_pairs(s1_keys, s1_entity_ids, other_df, src_name,
                            max_candidates_per_entity=40, flush_every=200_000):
    if not s1_entity_ids or len(other_df) == 0:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id",
                                      "cosine_sim", "src"])

    print(f"    Building blocking indexes for {src_name}: "
          f"{len(other_df):,} records", flush=True)

    prefix_index, token_index, number_index = _build_indexes(other_df)

    total = len(s1_entity_ids)
    buffer = []
    chunks = []

    for counter, eid in enumerate(s1_entity_ids, start=1):
        prefix, tokens, numbers = s1_keys[eid]

        candidates = _get_candidates(prefix, tokens, numbers,
                                      prefix_index, token_index, number_index)

        for candidate_id in list(candidates)[:max_candidates_per_entity]:
            buffer.append((eid, candidate_id, 0.0, src_name))

        # Flush the raw tuple buffer into a compact DataFrame periodically.
        # This bounds how large the Python-list-of-tuples ever gets: a list
        # of millions of (str, str, float, str) tuples is much heavier than
        # the same data held as typed pandas columns, so converting in
        # bounded chunks -- instead of accumulating one giant list for the
        # whole pass -- is a real reduction in peak memory, not just style.
        if counter % flush_every == 0:
            chunks.append(_pairs_list_to_df(buffer))
            print(f"      processed {counter:,}/{total:,} S1 records", flush=True)

    if buffer:
        chunks.append(_pairs_list_to_df(buffer))
    del buffer

    del prefix_index, token_index, number_index
    gc.collect()

    if not chunks:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id",
                                      "cosine_sim", "src"])

    result = pd.concat(chunks, ignore_index=True)
    del chunks
    gc.collect()
    return result


# ============================================================
# MAIN CANDIDATE GENERATION
# ============================================================

def _pairs_list_to_df(pairs_list):
    df = pd.DataFrame(pairs_list, columns=["source1_entity_id", "candidate_entity_id",
                                            "cosine_sim", "src"])
    pairs_list.clear()
    return df


def generate_candidates(source1_df, source2_df, source3_df, top_k=15,
                         min_sim=0.15, max_candidates_per_entity=40):
    """
    NOTE: top_k and min_sim are kept in the signature so train.py/predict.py
    don't need to change. This implementation does not use TF-IDF.
    """
    print("\nStarting memory-efficient blocking...", flush=True)

    print("  precomputing Source1 blocking keys once "
          "(reused for both the S2 and S3 passes)...", flush=True)
    s1_keys = _precompute_keys(source1_df)
    s1_entity_ids = list(s1_keys.keys())

    print("\nSource 1 -> Source 2", flush=True)
    df_s2 = _generate_source_pairs(s1_keys, s1_entity_ids, source2_df, "S2",
                                    max_candidates_per_entity=max_candidates_per_entity)

    print("\nSource 1 -> Source 3", flush=True)
    df_s3 = _generate_source_pairs(s1_keys, s1_entity_ids, source3_df, "S3",
                                    max_candidates_per_entity=max_candidates_per_entity)
    del s1_keys
    gc.collect()

    pairs_df = pd.concat([df_s2, df_s3], ignore_index=True)
    del df_s2, df_s3
    gc.collect()

    if len(pairs_df) == 0:
        return pd.DataFrame(columns=["source1_entity_id", "candidate_entity_id",
                                      "cosine_sim", "src"])

    pairs_df = pairs_df.drop_duplicates(subset=["source1_entity_id", "candidate_entity_id"])

    pairs_df = pairs_df.sort_values(["source1_entity_id", "candidate_entity_id"])
    pairs_df["_rank"] = pairs_df.groupby("source1_entity_id").cumcount()
    pairs_df = pairs_df[pairs_df["_rank"] < max_candidates_per_entity].drop(columns="_rank")
    pairs_df = pairs_df.reset_index(drop=True)

    print(f"\nCandidate generation complete: {len(pairs_df):,} candidate pairs", flush=True)

    return pairs_df


# ============================================================
# TSV OUTPUT
# ============================================================

def candidates_to_tsv_rows(source1_ids, pairs_df):
    if len(pairs_df) == 0:
        return pd.DataFrame(
            [{"source1_entity_id": s1_id, "candidate_entity_ids": ""} for s1_id in source1_ids]
        )

    grouped = pairs_df.groupby("source1_entity_id")["candidate_entity_id"].apply(
        lambda ids: ",".join(dict.fromkeys(ids))
    )


    rows = []

    for s1_id in source1_ids:
        rows.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": grouped.get(s1_id, ""),
        })
    return pd.DataFrame(rows)
