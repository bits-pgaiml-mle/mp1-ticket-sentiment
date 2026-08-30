# Dataset report — `sentiment140`

**Source mode:** `sentiment140`  
**Pipeline status:** PASS  
**Active raw file:** `data/raw/tickets.csv`

## Profile

| Metric | Value |
|--------|-------|
| Rows | 1500 |
| Unique texts | 105 |
| Avg text length | 32.4 |

### Label distribution

| Label | Count |
|-------|------:|
| negative | 500 |
| neutral | 501 |
| positive | 499 |

### Channel distribution

| Channel | Count |
|---------|------:|
| app | 526 |
| chat | 652 |
| email | 322 |

### data_source mix

| data_source | Count |
|-------------|------:|
| sentiment140 | 1500 |

## Validation log

```text
Prepared source=sentiment140 -> D:\TVSRAO\BITS\PGAIML\Course\ML Engineering\Mini-Project-1\mp1-ticket-sentiment\data\raw\tickets.csv (1500 rows)
data_source
sentiment140    1500

PASS: 1500 tickets validated â€” schema + statistical checks passed
label
neutral     501
negative    500
positive    499
```

## Training summary

- **Best run:** `logreg_C1`
- **Accuracy:** 1.0
- **Macro-F1:** 1.0
- **Justification:** Selected highest macro-F1 among LogisticRegression (C=1/C=10) and LinearSVC. TF-IDF is fit only on the training split to avoid leakage.

```text
Feature store: 1500 rows -> D:\TVSRAO\BITS\PGAIML\Course\ML Engineering\Mini-Project-1\mp1-ticket-sentiment\feature_store\feature_store.db [ticket_features]
Run logreg_C1: accuracy=1.0000 f1_macro=1.0000
Run logreg_C10: accuracy=1.0000 f1_macro=1.0000
Run linear_svc: accuracy=1.0000 f1_macro=1.0000
Best model: logreg_C1 -> D:\TVSRAO\BITS\PGAIML\Course\ML Engineering\Mini-Project-1\mp1-ticket-sentiment\model_store\sentiment_model.joblib
{
  "best_run": "logreg_C1",
  "metrics": {
    "accuracy": 1.0,
    "f1_macro": 1.0
  },
  "justification": "Selected highest macro-F1 among LogisticRegression (C=1/C=10) and LinearSVC. TF-IDF is fit only on the training split to avoid leakage."
```

## Reproduce

```bash
python data/prepare_dataset.py --source sentiment140
python validation/validate_data.py
python features/build_features.py
python training/train.py
```
