"""
Normalization utilities for business names and addresses -- OPTIMIZED.

Same normalization rules as before. The difference is HOW they're applied:
add_normalized_columns now runs each regex substitution once across the
WHOLE column (a vectorized pandas .str operation, executed in C) instead
of calling a Python function once per row via .apply(). Across three
sources totaling ~12.5 million rows, that's the difference between a
handful of bulk passes and 12.5 million individual Python function calls.

The single-string normalize_name()/normalize_address() functions are kept
for any ad-hoc/one-off use, built on the same replacement tables.
"""
import re

NAME_REPLACEMENTS = [
    (r"\bincorporated\b", "inc"),
    (r"\bcorporation\b", "corp"),
    (r"\bcompany\b", "co"),
    (r"\blimited\b", "ltd"),
    (r"\bpvt\.?\b", "private"),
    (r"\bpte\.?\b", "private"),
    (r"\bllp\b", "llp"),
    (r"\bllc\b", "llc"),
    (r"\b&\b", "and"),
    (r"\bdba\b", ""),
]

ADDRESS_REPLACEMENTS = [
    (r"\broad\b", "rd"),
    (r"\bstreet\b", "st"),
    (r"\bavenue\b", "ave"),
    (r"\bboulevard\b", "blvd"),
    (r"\bdrive\b", "dr"),
    (r"\blane\b", "ln"),
    (r"\bapartment\b", "apt"),
    (r"\bsuite\b", "ste"),
    (r"\bfloor\b", "fl"),
    (r"\bnear\b", ""),
    (r"\bnr\.?\b", ""),
    (r"\bopposite\b", ""),
    (r"\bopp\.?\b", ""),
]

# Precompiled once, shared by both the single-string and vectorized paths.
_NAME_REPLACEMENTS_COMPILED = [(re.compile(p), r) for p, r in NAME_REPLACEMENTS]
_ADDR_REPLACEMENTS_COMPILED = [(re.compile(p), r) for p, r in ADDRESS_REPLACEMENTS]
_NON_ALNUM_SPACE_RE = re.compile(r"[^a-z0-9\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")


# ============================================================
# Single-string versions (unchanged behavior, ad-hoc use)
# ============================================================

def _basic_clean(text):
    if text is None or isinstance(text, float):
        return ""
    text = str(text).lower().strip()
    text = text.replace("&", " and ")
    text = _NON_ALNUM_SPACE_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    return text


def normalize_name(name):
    text = _basic_clean(name)
    for pattern, repl in _NAME_REPLACEMENTS_COMPILED:
        text = pattern.sub(repl, text)
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def normalize_address(address):
    text = _basic_clean(address)
    for pattern, repl in _ADDR_REPLACEMENTS_COMPILED:
        text = pattern.sub(repl, text)
    return _MULTI_SPACE_RE.sub(" ", text).strip()


def normalize_country(country):
    return _basic_clean(country)


# ============================================================
# Vectorized versions -- used by add_normalized_columns
# ============================================================

def _basic_clean_series(s):
    s = s.fillna("").astype(str).str.lower().str.strip()
    s = s.str.replace("&", " and ", regex=False)
    s = s.str.replace(r"[^a-z0-9\s]", " ", regex=True)
    s = s.str.replace(r"\s+", " ", regex=True).str.strip()
    return s


def _normalize_name_series(s):
    s = _basic_clean_series(s)
    for pattern, repl in NAME_REPLACEMENTS:
        s = s.str.replace(pattern, repl, regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def _normalize_address_series(s):
    s = _basic_clean_series(s)
    for pattern, repl in ADDRESS_REPLACEMENTS:
        s = s.str.replace(pattern, repl, regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def _normalize_country_series(s):
    return _basic_clean_series(s)


def add_normalized_columns(df, name_col="business_name", addr_col="business_address",
                            country_col="country"):
    df = df.copy()
    df["name_norm"] = _normalize_name_series(df[name_col])
    df["addr_norm"] = _normalize_address_series(df[addr_col])
    df["country_norm"] = _normalize_country_series(df[country_col])
    return df