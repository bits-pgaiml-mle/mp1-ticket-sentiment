# External / Kaggle data drop zone (Flavor C)

## Modes (`configs/data_source.yaml` or `--source`)

| Mode | What it does |
|------|----------------|
| `synthetic` | Generated support tickets only (default) |
| `kaggle` | Adapt a reviews CSV into ticket schema |
| `both` | Concatenate synthetic tickets + Kaggle-adapted reviews |

## How to use real Kaggle data

1. Export a CSV from Amazon reviews, Yelp, Sentiment140, etc. with a text column and a rating/label column.
2. Save as `data/external/kaggle/reviews_sample.csv` (or set `kaggle.local_csv` in config).
3. Map columns in `configs/data_source.yaml` if needed (`text_col`, `label_col`).
4. Run:

```bash
python data/prepare_dataset.py --source kaggle
python data/prepare_dataset.py --source both
```

Star ratings are mapped: ≤2 → negative, 3 → neutral, ≥4 → positive.
If the CSV is missing, a tiny demo file is created so the pipeline still runs.
