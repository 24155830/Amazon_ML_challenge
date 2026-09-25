"""
Pairwise feature engineering for (source1_record, candidate_record) pairs.
All features are symmetric string-similarity / structural signals -- no
external lookups (per the challenge's fair-play rules).
"""
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler


def _token_set(s):
    return set(s.split())


def _jaccard(a, b):
    ta, tb = _token_set(a), _token_set(b)
    if not ta and not tb:
        return 1.0
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _first_token_match(a, b):
    ta, tb = a.split(), b.split()
    if not ta or not tb:
        return 0
    return int(ta[0] == tb[0])


def pair_features(name1, addr1, country1, name2, addr2, country2, cosine_sim=None):
    feats = {}

    # --- name features ---
    feats["name_levenshtein_ratio"] = Levenshtein.normalized_similarity(name1, name2)
    feats["name_jaro_winkler"] = JaroWinkler.similarity(name1, name2)
    feats["name_token_sort"] = fuzz.token_sort_ratio(name1, name2) / 100.0
    feats["name_token_set"] = fuzz.token_set_ratio(name1, name2) / 100.0
    feats["name_partial"] = fuzz.partial_ratio(name1, name2) / 100.0
    feats["name_jaccard"] = _jaccard(name1, name2)
    feats["name_first_token_match"] = _first_token_match(name1, name2)
    feats["name_len_diff"] = abs(len(name1) - len(name2))
    feats["name_len_ratio"] = (min(len(name1), len(name2)) + 1) / (max(len(name1), len(name2)) + 1)

    # --- address features ---
    feats["addr_levenshtein_ratio"] = Levenshtein.normalized_similarity(addr1, addr2)
    feats["addr_jaro_winkler"] = JaroWinkler.similarity(addr1, addr2)
    feats["addr_token_sort"] = fuzz.token_sort_ratio(addr1, addr2) / 100.0
    feats["addr_token_set"] = fuzz.token_set_ratio(addr1, addr2) / 100.0
    feats["addr_jaccard"] = _jaccard(addr1, addr2)

    # numeric tokens in address often carry building/PIN numbers -- overlap
    # of digit tokens is a strong precision signal
    digits1 = {t for t in addr1.split() if t.isdigit()}
    digits2 = {t for t in addr2.split() if t.isdigit()}
    if digits1 and digits2:
        feats["addr_digit_overlap"] = len(digits1 & digits2) / len(digits1 | digits2)
    else:
        feats["addr_digit_overlap"] = 0.5  # unknown / not informative

    # --- country ---
    feats["country_match"] = int(country1 == country2)

    # --- blocking-stage signal, reused as a feature ---
    feats["blocking_cosine_sim"] = cosine_sim if cosine_sim is not None else 0.0

    return feats


def build_feature_frame(pairs_df, s1_lookup, other_lookup):
    """
    pairs_df: columns source1_entity_id, candidate_entity_id, cosine_sim
    s1_lookup / other_lookup: dict entity_id -> (name_norm, addr_norm, country_norm)
    Returns a DataFrame of features aligned row-for-row with pairs_df.
    """
    rows = []
    for s1_id, cand_id, sim in zip(pairs_df["source1_entity_id"],
                                    pairs_df["candidate_entity_id"],
                                    pairs_df["cosine_sim"]):
        n1, a1, c1 = s1_lookup[s1_id]
        n2, a2, c2 = other_lookup[cand_id]
        rows.append(pair_features(n1, a1, c1, n2, a2, c2, cosine_sim=sim))
    feat_df = pd.DataFrame(rows)
    return feat_df


FEATURE_COLUMNS = [
    "name_levenshtein_ratio", "name_jaro_winkler", "name_token_sort", "name_token_set",
    "name_partial", "name_jaccard", "name_first_token_match", "name_len_diff", "name_len_ratio",
    "addr_levenshtein_ratio", "addr_jaro_winkler", "addr_token_sort", "addr_token_set",
    "addr_jaccard", "addr_digit_overlap", "country_match", "blocking_cosine_sim",
]
