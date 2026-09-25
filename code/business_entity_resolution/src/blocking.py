"""
Memory-efficient candidate generation.

Instead of building a huge TF-IDF matrix over millions of records,
this implementation uses cheap blocking keys based on:

1. Country
2. Business-name prefix
3. Business-name tokens
4. Address numeric tokens

Multiple blocking strategies are UNIONED together so that a match
can be discovered through more than one key.

The expensive fuzzy features are still calculated later by features.py.
"""

import re
import gc
from collections import defaultdict

import pandas as pd


# ============================================================
# BASIC HELPERS
# ============================================================

def _tokens(text):
    if text is None:
        return []

    text = str(text).strip()

    if not text:
        return []

    return text.split()


def _name_prefix(name, n=4):
    """
    First n characters of normalized business name.
    """
    if not name:
        return ""

    cleaned = re.sub(r"[^a-z0-9]", "", str(name).lower())

    return cleaned[:n]


def _name_tokens(name):
    """
    Useful words from a normalized business name.

    Very short tokens are ignored because words like:
    'a', 'of', 'in' are weak blocking signals.
    """
    if not name:
        return set()

    tokens = set()

    for token in str(name).split():

        if len(token) >= 3:
            tokens.add(token)

    return tokens


def _address_numbers(address):
    """
    Extract numeric pieces from an address.

    Examples:

        '17560 ellis road' -> {'17560'}
        '1/95 westchester drive' -> {'1', '95'}
        'kh no 570 1' -> {'570', '1'}
    """

    if not address:
        return set()

    return set(re.findall(r"\d+", str(address)))


# ============================================================
# BUILD INDEXES
# ============================================================

def _build_indexes(other_df):

    prefix_index = defaultdict(list)
    token_index = defaultdict(list)
    number_index = defaultdict(list)

    for row in other_df.itertuples():

        entity_id = row.entity_id

        # -------------------------
        # Name prefix
        # -------------------------

        prefix = _name_prefix(row.name_norm)

        if prefix:
            prefix_index[prefix].append(entity_id)

        # -------------------------
        # Name tokens
        # -------------------------

        for token in _name_tokens(row.name_norm):
            token_index[token].append(entity_id)

        # -------------------------
        # Address numbers
        # -------------------------

        for number in _address_numbers(row.addr_norm):
            number_index[number].append(entity_id)

    return prefix_index, token_index, number_index


# ============================================================
# CANDIDATES FOR ONE RECORD
# ============================================================

def _get_candidates(
    s1_row,
    prefix_index,
    token_index,
    number_index,
    max_from_each_key=15,
):
    """
    Generate candidates for one Source-1 record.

    We deliberately limit the number obtained from each blocking key.
    """

    candidates = set()

    # --------------------------------------------------------
    # 1. Name prefix
    # --------------------------------------------------------

    prefix = _name_prefix(s1_row.name_norm)

    if prefix:

        ids = prefix_index.get(prefix, [])

        candidates.update(ids[:max_from_each_key])

    # --------------------------------------------------------
    # 2. Name tokens
    # --------------------------------------------------------

    name_tokens = _name_tokens(s1_row.name_norm)

    for token in name_tokens:

        ids = token_index.get(token, [])

        candidates.update(ids[:max_from_each_key])

    # --------------------------------------------------------
    # 3. Address numbers
    # --------------------------------------------------------

    numbers = _address_numbers(s1_row.addr_norm)

    for number in numbers:

        ids = number_index.get(number, [])

        candidates.update(ids[:max_from_each_key])

    return candidates


# ============================================================
# GENERATE CANDIDATES BETWEEN TWO SOURCES
# ============================================================

def _generate_source_pairs(
    s1_df,
    other_df,
    src_name,
    max_candidates_per_entity=40,
):
    """
    Generate candidates from Source-1 to one other source.
    """

    if len(s1_df) == 0 or len(other_df) == 0:
        return []

    print(
        f"    Building blocking indexes for {src_name}: "
        f"{len(other_df):,} records"
    )

    (
        prefix_index,
        token_index,
        number_index,
    ) = _build_indexes(other_df)

    results = []

    total = len(s1_df)

    for counter, row in enumerate(
        s1_df.itertuples(),
        start=1,
    ):

        candidates = _get_candidates(
            row,
            prefix_index,
            token_index,
            number_index,
        )

        # Limit total candidates per S1 entity.
        candidates = list(candidates)[
            :max_candidates_per_entity
        ]

        for candidate_id in candidates:

            results.append(
                (
                    row.entity_id,
                    candidate_id,
                    0.0,
                    src_name,
                )
            )

        # Progress every 50,000 records.
        if counter % 50000 == 0:

            print(
                f"      processed "
                f"{counter:,}/{total:,} S1 records"
            )

    # Free indexes before moving to next source.
    del prefix_index
    del token_index
    del number_index

    gc.collect()

    return results


# ============================================================
# MAIN CANDIDATE GENERATION
# ============================================================

def generate_candidates(
    source1_df,
    source2_df,
    source3_df,
    top_k=15,
    min_sim=0.15,
    max_candidates_per_entity=40,
):
    """
    Generate memory-efficient candidate pairs.

    NOTE:
    top_k and min_sim are kept in the function signature so that
    train.py and predict.py do not need to be changed.

    This implementation does not use TF-IDF.
    """

    print("\nStarting memory-efficient blocking...")

    all_pairs = []

    # ========================================================
    # SOURCE 1 -> SOURCE 2
    # ========================================================

    print("\nSource 1 -> Source 2")

    pairs_s2 = _generate_source_pairs(
        source1_df,
        source2_df,
        "S2",
        max_candidates_per_entity=max_candidates_per_entity,
    )

    all_pairs.extend(pairs_s2)

    del pairs_s2

    gc.collect()

    # ========================================================
    # SOURCE 1 -> SOURCE 3
    # ========================================================

    print("\nSource 1 -> Source 3")

    pairs_s3 = _generate_source_pairs(
        source1_df,
        source3_df,
        "S3",
        max_candidates_per_entity=max_candidates_per_entity,
    )

    all_pairs.extend(pairs_s3)

    del pairs_s3

    gc.collect()

    # ========================================================
    # DATAFRAME
    # ========================================================

    pairs_df = pd.DataFrame(
        all_pairs,
        columns=[
            "source1_entity_id",
            "candidate_entity_id",
            "cosine_sim",
            "src",
        ],
    )

    del all_pairs

    gc.collect()

    # ========================================================
    # DEDUPLICATE
    # ========================================================

    if len(pairs_df) == 0:

        return pd.DataFrame(
            columns=[
                "source1_entity_id",
                "candidate_entity_id",
                "cosine_sim",
                "src",
            ]
        )

    pairs_df = pairs_df.drop_duplicates(
        subset=[
            "source1_entity_id",
            "candidate_entity_id",
        ]
    )

    # ========================================================
    # CAP CANDIDATES
    # ========================================================

    pairs_df = pairs_df.sort_values(
        [
            "source1_entity_id",
            "candidate_entity_id",
        ]
    )

    pairs_df["_rank"] = (
        pairs_df
        .groupby("source1_entity_id")
        .cumcount()
    )

    pairs_df = pairs_df[
        pairs_df["_rank"] < max_candidates_per_entity
    ].drop(
        columns="_rank"
    )

    pairs_df = pairs_df.reset_index(
        drop=True
    )

    print(
        f"\nCandidate generation complete:"
        f" {len(pairs_df):,} candidate pairs"
    )

    return pairs_df


# ============================================================
# TSV OUTPUT
# ============================================================

def candidates_to_tsv_rows(
    source1_ids,
    pairs_df,
):
    """
    Convert candidate pairs into the required format:

    source1_entity_id    candidate_entity_ids
    """

    if len(pairs_df) == 0:

        return pd.DataFrame(
            [
                {
                    "source1_entity_id": s1_id,
                    "candidate_entity_ids": "",
                }
                for s1_id in source1_ids
            ]
        )

    grouped = (
        pairs_df
        .groupby("source1_entity_id")[
            "candidate_entity_id"
        ]
        .apply(
            lambda ids: ",".join(
                dict.fromkeys(ids)
            )
        )
    )

    rows = []

    for s1_id in source1_ids:

        rows.append(
            {
                "source1_entity_id": s1_id,
                "candidate_entity_ids": grouped.get(
                    s1_id,
                    "",
                ),
            }
        )

    return pd.DataFrame(rows)