# Business Entity Resolution — Pipeline

Matches noisy business records across Source 1 (reference), Source 2, and
Source 3 using TF-IDF/cosine blocking + a LightGBM binary classifier over
string-similarity features, with the decision threshold tuned directly
against the macro-averaged F_0.5 metric used for scoring.

No external data, APIs, or lookups are used anywhere in the pipeline —
every feature is computed from the provided fields only.

## 1. Setup

```bash
cd code/business_entity_resolution
python3 -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt
```

## 2. Expected data layout

Place the challenge's `dataset/` folder at the same level as `code/` and
`output/` (i.e. matching the submission zip structure), or pass `--data-dir`
explicitly:

```
dataset/
  train/
    train_source1.tsv
    train_source2.tsv
    train_source3.tsv
    train_ground_truth.tsv
  test/
    test_source1.tsv
    test_source2.tsv
    test_source3.tsv
```

## 3. Train

```bash
python3 src/train.py --data-dir ../../dataset --out-dir ../../artifacts
```

This will:
- normalize names/addresses,
- run blocking on the training data and print the **recall ceiling**
  (fraction of true matches that survive blocking — check this first;
  if it's low, loosen blocking in `src/blocking.py` before touching the
  model),
- label candidate pairs against `train_ground_truth.tsv`,
- group-split by `source1_entity_id` (a S1 entity's pairs never span
  both train and validation — this avoids leakage),
- train a LightGBM classifier,
- sweep the decision threshold on the validation split to maximize
  macro-averaged F_0.5 (the actual leaderboard metric — not AUC/accuracy),
- save `artifacts/model.joblib` and `artifacts/config.json`.

Console output reports the validation F_0.5 you should expect on the
public leaderboard.

## 4. Predict on test set

```bash
python3 src/predict.py --data-dir ../../dataset --artifacts-dir ../../artifacts \
    --out-dir ../../output
```

Writes `output/candidate_pairs.tsv` and `output/matching_results.tsv`.

## 5. Validate before submitting

```bash
cd ../..   # back to student_resource/ root
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

Fix anything it flags, then upload `output/matching_results.tsv` to the
Portal.

## Pipeline design

**Blocking (`src/blocking.py`)** — buckets records by normalized country,
then within each bucket fits a character n-gram (3–5) TF-IDF vectorizer
over `name + address` text and pulls the top-K cosine-nearest S2/S3
records for every S1 record. A second, unblocked fallback pass re-searches
across *all* records (ignoring country) for any S1 entity that got fewer
than 3 candidates from its own country bucket — this protects recall
against noisy/missing/mislabeled country fields, and against countries
(like France in the test set) that were never seen during training.

**Features (`src/features.py`)** — Levenshtein ratio, Jaro-Winkler, token
sort/set ratio, and Jaccard token overlap on both name and address;
digit-token overlap on address (catches matching building/PIN numbers
even when everything else in the address is reworded); exact country
match; and the blocking-stage cosine similarity, reused as a feature
rather than thrown away.

**Model (`src/train.py`)** — LightGBM binary classifier (MIT-licensed,
a few hundred KB, nowhere near the 8B-parameter cap), `scale_pos_weight`
to handle the heavy class imbalance (most candidate pairs are non-matches).
Threshold is chosen by directly sweeping macro-F_0.5 on a held-out
validation split, not a generic 0.5 cutoff — this matters a lot for a
precision-weighted metric.

## Tuning for a stronger score

- Raise `top_k` / lower `min_sim` in `blocking.py` if the printed recall
  ceiling is below ~0.97 — you cannot out-model a low blocking recall.
- Add more `FEATURE_COLUMNS` in `features.py` (e.g. Soundex/phonetic
  match on the first name token, city/PIN extraction from address) and
  re-run `train.py`.
- Because F_0.5 penalizes false positives 2x, err toward a **higher**
  threshold if validation precision looks weak per source1 entity.
