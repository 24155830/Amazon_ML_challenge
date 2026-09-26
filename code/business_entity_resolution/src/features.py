"""
Pairwise feature engineering -- OPTIMIZED.

Same features, same math as before. The only change is HOW the final
DataFrame is assembled: build_feature_frame now fills pre-allocated
per-column lists directly, instead of building one dict per row and
handing pandas.DataFrame() a list of millions of dicts (a well-known slow
path -- pandas has to infer structure from every dict individually).
Filling columns directly and building the DataFrame once from a dict of
columns is substantially faster at this scale (candidate sets can run into
the tens of millions of rows).
"""
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import Levenshtein, JaroWinkler

FEATURE_COLUMNS = [
    "name_levenshtein_ratio", "name_jaro_winkler", "name_token_sort", "name_token_set",
    "name_partial", "name_jaccard", "name_first_token_match", "name_len_diff", "name_len_ratio",
    "addr_levenshtein_ratio", "addr_jaro_winkler", "addr_token_sort", "addr_token_set",
    "addr_jaccard", "addr_digit_overlap", "country_match", "blocking_cosine_sim",
]


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
    """Kept for single-pair / ad-hoc use; returns a dict as before."""
    feats = {}

    feats["name_levenshtein_ratio"] = Levenshtein.normalized_similarity(name1, name2)
    feats["name_jaro_winkler"] = JaroWinkler.similarity(name1, name2)
    feats["name_token_sort"] = fuzz.token_sort_ratio(name1, name2) / 100.0
    feats["name_token_set"] = fuzz.token_set_ratio(name1, name2) / 100.0
    feats["name_partial"] = fuzz.partial_ratio(name1, name2) / 100.0
    feats["name_jaccard"] = _jaccard(name1, name2)
    feats["name_first_token_match"] = _first_token_match(name1, name2)
    feats["name_len_diff"] = abs(len(name1) - len(name2))
    feats["name_len_ratio"] = (min(len(name1), len(name2)) + 1) / (max(len(name1), len(name2)) + 1)

    feats["addr_levenshtein_ratio"] = Levenshtein.normalized_similarity(addr1, addr2)
    feats["addr_jaro_winkler"] = JaroWinkler.similarity(addr1, addr2)
    feats["addr_token_sort"] = fuzz.token_sort_ratio(addr1, addr2) / 100.0
    feats["addr_token_set"] = fuzz.token_set_ratio(addr1, addr2) / 100.0
    feats["addr_jaccard"] = _jaccard(addr1, addr2)

    digits1 = {t for t in addr1.split() if t.isdigit()}
    digits2 = {t for t in addr2.split() if t.isdigit()}
    if digits1 and digits2:
        feats["addr_digit_overlap"] = len(digits1 & digits2) / len(digits1 | digits2)
    else:
        feats["addr_digit_overlap"] = 0.5

    feats["country_match"] = int(country1 == country2)
    feats["blocking_cosine_sim"] = cosine_sim if cosine_sim is not None else 0.0

    return feats


def build_feature_frame(pairs_df, s1_lookup, other_lookup):
    """
    pairs_df: columns source1_entity_id, candidate_entity_id, cosine_sim
    s1_lookup / other_lookup: dict entity_id -> (name_norm, addr_norm, country_norm)

    Returns a DataFrame of features aligned row-for-row with pairs_df.
    Builds columns directly (dict of lists -> one DataFrame.from_dict call)
    instead of a list of per-row dicts, which is much faster to assemble
    at millions of rows.
    """
    n = len(pairs_df)
    cols = {name: [None] * n for name in FEATURE_COLUMNS}

    s1_ids = pairs_df["source1_entity_id"].values
    cand_ids = pairs_df["candidate_entity_id"].values
    sims = pairs_df["cosine_sim"].values

    # local references -- avoids repeated attribute lookups in the hot loop
    lev_sim = Levenshtein.normalized_similarity
    jw_sim = JaroWinkler.similarity
    tsort = fuzz.token_sort_ratio
    tset = fuzz.token_set_ratio
    partial = fuzz.partial_ratio
    jaccard = _jaccard
    first_tok = _first_token_match

    c_name_lev = cols["name_levenshtein_ratio"]
    c_name_jw = cols["name_jaro_winkler"]
    c_name_tsort = cols["name_token_sort"]
    c_name_tset = cols["name_token_set"]
    c_name_partial = cols["name_partial"]
    c_name_jac = cols["name_jaccard"]
    c_name_first = cols["name_first_token_match"]
    c_name_lendiff = cols["name_len_diff"]
    c_name_lenratio = cols["name_len_ratio"]
    c_addr_lev = cols["addr_levenshtein_ratio"]
    c_addr_jw = cols["addr_jaro_winkler"]
    c_addr_tsort = cols["addr_token_sort"]
    c_addr_tset = cols["addr_token_set"]
    c_addr_jac = cols["addr_jaccard"]
    c_addr_digit = cols["addr_digit_overlap"]
    c_country = cols["country_match"]
    c_block = cols["blocking_cosine_sim"]

    for i in range(n):
        n1, a1, c1 = s1_lookup[s1_ids[i]]
        n2, a2, c2 = other_lookup[cand_ids[i]]

        c_name_lev[i] = lev_sim(n1, n2)
        c_name_jw[i] = jw_sim(n1, n2)
        c_name_tsort[i] = tsort(n1, n2) / 100.0
        c_name_tset[i] = tset(n1, n2) / 100.0
        c_name_partial[i] = partial(n1, n2) / 100.0
        c_name_jac[i] = jaccard(n1, n2)
        c_name_first[i] = first_tok(n1, n2)
        c_name_lendiff[i] = abs(len(n1) - len(n2))
        c_name_lenratio[i] = (min(len(n1), len(n2)) + 1) / (max(len(n1), len(n2)) + 1)

        c_addr_lev[i] = lev_sim(a1, a2)
        c_addr_jw[i] = jw_sim(a1, a2)
        c_addr_tsort[i] = tsort(a1, a2) / 100.0
        c_addr_tset[i] = tset(a1, a2) / 100.0
        c_addr_jac[i] = jaccard(a1, a2)

        digits1 = {t for t in a1.split() if t.isdigit()}
        digits2 = {t for t in a2.split() if t.isdigit()}
        c_addr_digit[i] = (len(digits1 & digits2) / len(digits1 | digits2)
                           if (digits1 and digits2) else 0.5)

        c_country[i] = int(c1 == c2)
        c_block[i] = sims[i] if sims[i] is not None else 0.0

    return pd.DataFrame(cols)