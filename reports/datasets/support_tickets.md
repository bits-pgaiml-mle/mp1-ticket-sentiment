# Dataset report — `support_tickets`

**Source mode:** `support_tickets`  
**Pipeline status:** PASS  
**Active raw file:** `data/raw/tickets.csv`

## Profile

| Metric | Value |
|--------|-------|
| Rows | 1500 |
| Unique texts | 1463 |
| Avg text length | 107.8 |

### Label distribution

| Label | Count |
|-------|------:|
| negative | 496 |
| neutral | 524 |
| positive | 480 |

### Channel distribution

| Channel | Count |
|---------|------:|
| app | 470 |
| chat | 522 |
| email | 508 |

### data_source mix

| data_source | Count |
|-------------|------:|
| support_tickets | 1500 |

## Validation log

```text
Prepared source=support_tickets -> D:\TVSRAO\BITS\PGAIML\Course\ML Engineering\Mini-Project-1\mp1-ticket-sentiment\data\raw\tickets.csv (1500 rows)
data_source
support_tickets    1500

PASS: 1500 tickets validated â€” schema + statistical checks passed
label
neutral     524
negative    496
positive    480
```

## Training summary

- **Best run:** `linear_svc`
- **Accuracy:** 0.8433333333333334
- **Macro-F1:** 0.8422550258817864
- **Justification:** Selected highest macro-F1 among LogisticRegression (C=1/C=10) and LinearSVC. TF-IDF is fit only on the training split to avoid leakage.

```text
Feature store: 1500 rows -> D:\TVSRAO\BITS\PGAIML\Course\ML Engineering\Mini-Project-1\mp1-ticket-sentiment\feature_store\feature_store.db [ticket_features]
Run logreg_C1: accuracy=0.8267 f1_macro=0.8251
Run logreg_C10: accuracy=0.8267 f1_macro=0.8257
Run linear_svc: accuracy=0.8433 f1_macro=0.8423
Best model: linear_svc -> D:\TVSRAO\BITS\PGAIML\Course\ML Engineering\Mini-Project-1\mp1-ticket-sentiment\model_store\sentiment_model.joblib
{
  "best_run": "linear_svc",
  "metrics": {
    "accuracy": 0.8433333333333334,
    "f1_macro": 0.8422550258817864
  },
  "justification": "Selected highest macro-F1 among LogisticRegression (C=1/C=10) and LinearSVC. TF-IDF is fit only on the training split to avoid leakage."
```

## Reproduce

```bash
python data/prepare_dataset.py --source support_tickets
python validation/validate_data.py
python features/build_features.py
python training/train.py
```
