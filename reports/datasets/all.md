# Dataset report — `all`

**Source mode:** `all`  
**Pipeline status:** PASS  
**Active raw file:** `data/raw/tickets.csv`

## Profile

| Metric | Value |
|--------|-------|
| Rows | 3200 |
| Unique texts | 1101 |
| Avg text length | 59.2 |

### Label distribution

| Label | Count |
|-------|------:|
| negative | 1059 |
| neutral | 1111 |
| positive | 1030 |

### Channel distribution

| Channel | Count |
|---------|------:|
| app | 1306 |
| chat | 1086 |
| email | 808 |

### data_source mix

| data_source | Count |
|-------------|------:|
| amazon | 800 |
| sentiment140 | 800 |
| support_tickets | 800 |
| yelp | 800 |

## Validation log

```text
Prepared source=all -> D:\TVSRAO\BITS\PGAIML\Course\ML Engineering\Mini-Project-1\mp1-ticket-sentiment\data\raw\tickets.csv (3200 rows)
data_source
amazon             800
yelp               800
sentiment140       800
support_tickets    800

PASS: 3200 tickets validated â€” schema + statistical checks passed
label
neutral     1111
negative    1059
positive    1030
```

## Training summary

- **Best run:** `linear_svc`
- **Accuracy:** 0.959375
- **Macro-F1:** 0.9590532183625009
- **Justification:** Selected highest macro-F1 among LogisticRegression (C=1/C=10) and LinearSVC. TF-IDF is fit only on the training split to avoid leakage.

```text
Feature store: 3200 rows -> D:\TVSRAO\BITS\PGAIML\Course\ML Engineering\Mini-Project-1\mp1-ticket-sentiment\feature_store\feature_store.db [ticket_features]
Run logreg_C1: accuracy=0.9469 f1_macro=0.9464
Run logreg_C10: accuracy=0.9547 f1_macro=0.9544
Run linear_svc: accuracy=0.9594 f1_macro=0.9591
Best model: linear_svc -> D:\TVSRAO\BITS\PGAIML\Course\ML Engineering\Mini-Project-1\mp1-ticket-sentiment\model_store\sentiment_model.joblib
{
  "best_run": "linear_svc",
  "metrics": {
    "accuracy": 0.959375,
    "f1_macro": 0.9590532183625009
  },
  "justification": "Selected highest macro-F1 among LogisticRegression (C=1/C=10) and LinearSVC. TF-IDF is fit only on the training split to avoid leakage."
```

## Reproduce

```bash
python data/prepare_dataset.py --source all
python validation/validate_data.py
python features/build_features.py
python training/train.py
```
