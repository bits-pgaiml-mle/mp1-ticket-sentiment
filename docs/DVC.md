# DVC usage — mp1-ticket-sentiment

## What is versioned

| Path | Sources |
|------|---------|
| `data/versions/amazon/` | Amazon reviews → tickets |
| `data/versions/yelp/` | Yelp reviews → tickets |
| `data/versions/sentiment140/` | Twitter Sentiment140 → tickets |
| `data/versions/support_tickets/` | Support-ticket / synthetic tickets |
| `data/versions/all/` | Mix of all four |
| `data/raw/tickets.csv` | Active dataset (from `configs/data_source.yaml`) |

Tracked via `dvc.yaml` stage `snapshot_datasets` (see `dvc.lock`).

## Setup

```bash
pip install -r requirements.txt
dvc pull   # if local remote has been pushed; otherwise regenerate below
```

## Regenerate and version all sources

```bash
dvc repro
# or: python scripts/snapshot_datasets.py
git add dvc.yaml dvc.lock .dvc .gitignore data/.gitignore
git commit -m "Update DVC dataset snapshots"
git tag -f week1-data-v1
```

## Switch active source without full repro

```bash
python data/prepare_dataset.py --source yelp
```

## After snapshot — run the M2 pipeline

```bash
python validation/validate_data.py
python features/build_features.py
```

This writes `feature_store/feature_store.db` and `model_store/feature_columns.json` from the active `data/raw/tickets.csv`.

```bash
dvc remote add -d localremote ./dvc-storage
dvc push
```

`dvc-storage/` is gitignored; teammates without the remote can run `dvc repro`.
