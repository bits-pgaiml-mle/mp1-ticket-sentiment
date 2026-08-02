# DVC usage — mp1-ticket-sentiment

## What is versioned

| Path | Sources |
|------|---------|
| `data/versions/amazon/` | Amazon reviews → tickets |
| `data/versions/yelp/` | Yelp reviews → tickets |
| `data/versions/sentiment140/` | Twitter Sentiment140 → tickets |
| `data/versions/support_tickets/` | Support-ticket / synthetic tickets |
| `data/versions/all/` | Mix of all four |
| `data/raw/tickets.csv` | Active dataset (from config) |

## Commands

```bash
dvc repro
git add dvc.yaml dvc.lock .dvc .gitignore
git commit -m "Update DVC dataset snapshots"
git tag -f week1-data-v1
```

Optional local remote:

```bash
dvc remote add -d localremote ./dvc-storage
dvc push
```
