import re
from typing import Dict, List

import numpy as np
import pandas as pd


DEFAULT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "had", "has", "have", "he", "her", "his", "i", "if", "in", "is",
    "it", "its", "me", "my", "of", "on", "or", "our", "she", "so", "that",
    "the", "their", "them", "there", "these", "they", "this", "to", "was",
    "we", "were", "with", "you", "your",
}


def clean_text(text: object, stopwords: set[str] | None = None) -> str:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""

    stopwords = stopwords or DEFAULT_STOPWORDS
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return ""

    tokens = [token for token in text.split() if token not in stopwords]
    return " ".join(tokens)


def prepare_interactions(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["user_id", "item_id", "rating", "time"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    work_df = df.copy()
    work_df["user_id"] = work_df["user_id"].astype(str)
    work_df["item_id"] = work_df["item_id"].astype(str)
    work_df = work_df.dropna(subset=["user_id", "item_id", "rating", "time"])
    work_df["rating"] = work_df["rating"].astype(float)
    work_df["time"] = pd.to_numeric(work_df["time"], errors="coerce")
    work_df = work_df.dropna(subset=["time"])
    work_df["time"] = work_df["time"].astype(np.int64)
    work_df = work_df.sort_values(["user_id", "time", "item_id"]).reset_index(drop=True)
    return work_df


def build_item_corpus(df: pd.DataFrame, max_reviews: int = 5) -> pd.DataFrame:
    text_cols = [col for col in ["title", "description", "review_text"] if col in df.columns]
    if "item_id" not in df.columns:
        raise ValueError("item_id column is required")

    rows: List[Dict[str, str]] = []
    for item_id, group in df.groupby("item_id", sort=False):
        title_parts = []
        desc_parts = []
        review_parts = []

        if "title" in text_cols:
            title_parts = [clean_text(x) for x in group["title"].dropna().tolist()]
        if "description" in text_cols:
            desc_parts = [clean_text(x) for x in group["description"].dropna().tolist()]
        if "review_text" in text_cols:
            review_parts = [clean_text(x) for x in group["review_text"].dropna().tolist()[:max_reviews]]

        merged_text = " ".join(part for part in [
            next((x for x in title_parts if x), ""),
            next((x for x in desc_parts if x), ""),
            " ".join(x for x in review_parts if x),
        ] if part).strip()

        image_path = ""
        if "image_path" in group.columns:
            image_path = next((str(x).strip() for x in group["image_path"].tolist() if pd.notna(x) and str(x).strip()), "")

        image_url = ""
        if "image_url" in group.columns:
            image_url = next((str(x).strip() for x in group["image_url"].tolist() if pd.notna(x) and str(x).strip()), "")

        rows.append({
            "item_id": str(item_id),
            "text": merged_text or "[EMPTY]",
            "image_path": image_path,
            "image_url": image_url,
        })

    return pd.DataFrame(rows)


def build_sequence_examples(interactions: pd.DataFrame, max_seq_len: int = 20, min_history_len: int = 1) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for user_id, group in interactions.groupby("user_id", sort=False):
        group = group.sort_values("time").reset_index(drop=True)
        history_items: List[str] = []
        history_ratings: List[float] = []

        for _, row in group.iterrows():
            if len(history_items) >= min_history_len:
                rows.append({
                    "user_id": str(user_id),
                    "item_id": str(row["item_id"]),
                    "rating": float(row["rating"]),
                    "time": int(row["time"]),
                    "history_item_ids": history_items[-max_seq_len:].copy(),
                    "history_ratings": history_ratings[-max_seq_len:].copy(),
                })

            history_items.append(str(row["item_id"]))
            history_ratings.append(float(row["rating"]))

    return pd.DataFrame(rows)
