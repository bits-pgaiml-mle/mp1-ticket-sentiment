# Dataset report — `amazon`

**Source mode:** `amazon`  
**Pipeline status:** PASS  
**Active raw file:** `data/raw/tickets.csv`

## Profile

| Metric | Value |
|--------|-------|
| Rows | 1500 |
| Unique texts | 105 |
| Avg text length | 50.1 |

### Label distribution

| Label | Count |
|-------|------:|
| negative | 486 |
| neutral | 514 |
| positive | 500 |

### Channel distribution

| Channel | Count |
|---------|------:|
| app | 742 |
| chat | 436 |
| email | 322 |

### data_source mix

| data_source | Count |
|-------------|------:|
| amazon | 1500 |

## Validation log

```text
Prepared source=amazon -> D:\TVSRAO\BITS\PGAIML\Course\ML Engineering\Mini-Project-1\mp1-ticket-sentiment\data\raw\tickets.csv (1500 rows)
data_source
amazon    1500

PASS: 1500 tickets validated â€” schema + statistical checks passed
label
neutral     514
positive    500
negative    486
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
python data/prepare_dataset.py --source amazon
python validation/validate_data.py
python features/build_features.py
python training/train.py
```
