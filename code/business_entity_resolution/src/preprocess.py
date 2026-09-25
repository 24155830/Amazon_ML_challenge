"""
Normalization utilities for business names and addresses.

The goal is NOT to destroy information, just to collapse superficial
variation (abbreviations, punctuation, casing) so that downstream string
similarity features are computed on comparable text.
"""
import re

# Applied in order. Each is a whole-word regex -> canonical replacement.
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
    (r"\bdba\b", ""),  # "doing business as" marker carries little signal alone
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


def _basic_clean(text: str) -> str:
    if text is None or (isinstance(text, float)):
        return ""
    text = str(text).lower().strip()
    # keep alnum, spaces, & (handled by replacement), drop other punctuation
    text = text.replace("&", " and ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_name(name: str) -> str:
    text = _basic_clean(name)
    for pattern, repl in NAME_REPLACEMENTS:
        text = re.sub(pattern, repl, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_address(address: str) -> str:
    text = _basic_clean(address)
    for pattern, repl in ADDRESS_REPLACEMENTS:
        text = re.sub(pattern, repl, text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_country(country: str) -> str:
    return _basic_clean(country)


def add_normalized_columns(df, name_col="business_name", addr_col="business_address",
                            country_col="country"):
    df = df.copy()
    df["name_norm"] = df[name_col].apply(normalize_name)
    df["addr_norm"] = df[addr_col].apply(normalize_address)
    df["country_norm"] = df[country_col].apply(normalize_country)
    return df
