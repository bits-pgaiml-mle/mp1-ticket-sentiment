import re

import pandas as pd


def clean_text(text: str) -> str:
    text = str(text).lower().strip()
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def channel_flags(channel: str) -> dict[str, int]:
    channel = str(channel).lower().strip()
    return {
        "channel_email": int(channel == "email"),
        "channel_chat": int(channel == "chat"),
        "channel_app": int(channel == "app"),
    }


def select_text(df: pd.DataFrame):
    return df["text_clean"].astype(str)


def select_numeric(df: pd.DataFrame):
    cols = ["text_len", "word_count", "channel_email", "channel_chat", "channel_app"]
    return df[cols]
