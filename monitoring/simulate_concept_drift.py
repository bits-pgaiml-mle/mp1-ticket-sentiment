import sys
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "config.yaml"

DRIFT_TICKETS = [
    {"text": "bruh this refund is taking forever lol", "channel": "chat"},
    {"text": "yo app glitched during checkout again fr", "channel": "app"},
    {"text": "need status on my tkt asap no cap", "channel": "chat"},
    {"text": "lowkey impressed by how fast support replied", "channel": "email"},
    {"text": "sus charge on my card, pls reverse", "channel": "email"},
    {"text": "delivery eta wrong again, this is wild", "channel": "app"},
    {"text": "can u share invoice copy for last order", "channel": "email"},
    {"text": "shipping slot change request for tomorrow", "channel": "chat"},
]


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    cfg = load_config()
    url = f"http://127.0.0.1:{cfg['serving']['port']}/predict"
    ok = 0
    for item in DRIFT_TICKETS:
        try:
            resp = requests.post(url, json=item, timeout=10)
            resp.raise_for_status()
            print(item["text"][:48], "->", resp.json())
            ok += 1
        except Exception as exc:
            print("Failed:", item["text"][:48], exc)
            sys.exit(1)
    print(f"Logged {ok} concept-drift style tickets. Next: python monitoring/check_drift.py")


if __name__ == "__main__":
    main()
