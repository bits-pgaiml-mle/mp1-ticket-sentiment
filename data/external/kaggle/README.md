# External data drop zone (Flavor C)

## Modes (`configs/data_source.yaml` or `--source`)

| Mode | Dataset |
|------|---------|
| `amazon` | Amazon product reviews (stars → sentiment) |
| `yelp` | Yelp reviews (stars → sentiment) |
| `sentiment140` | [Twitter Sentiment140](https://www.kaggle.com/datasets/kazanova/sentiment140) (0/2/4 → neg/neu/pos) |
| `support_tickets` | Support-ticket / intent-style text (uses built-in synthetic tickets if CSV missing) |
| `all` | Concatenate all four |

Aliases: `synthetic`/`support` → `support_tickets`, `twitter` → `sentiment140`, `both` → `all`.

## Drop-in paths

| Source | Default path | Typical columns |
|--------|--------------|-----------------|
| Amazon | `data/external/kaggle/amazon/reviews.csv` | `reviewText`, `overall` |
| Yelp | `data/external/kaggle/yelp/reviews.csv` | `text`, `stars` |
| Sentiment140 | `data/external/kaggle/sentiment140/tweets.csv` | no header: target,id,date,flag,user,text |
| Support tickets | `data/external/kaggle/support_tickets/tickets.csv` | `text`, `label` |

Column names are configurable under each section in `configs/data_source.yaml`.

## Commands

```bash
python data/prepare_dataset.py --source amazon
python data/prepare_dataset.py --source yelp
python data/prepare_dataset.py --source sentiment140
python data/prepare_dataset.py --source support_tickets
python data/prepare_dataset.py --source all
```

Demo CSVs are auto-created when files are missing so the pipeline still runs end-to-end.
