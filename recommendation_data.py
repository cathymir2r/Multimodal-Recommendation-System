import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from experiment_config import ExperimentConfig
from preprocessing import build_sequence_examples, prepare_interactions


def load_feature_dict(ids_path: str, feat_path: str) -> Dict[str, np.ndarray]:
    if not (os.path.exists(ids_path) and os.path.exists(feat_path)):
        raise FileNotFoundError(f"Missing feature files: {ids_path} or {feat_path}")

    item_ids = [str(x) for x in np.load(ids_path, allow_pickle=True)]
    features = np.load(feat_path)
    return {item_id: feature.astype(np.float32) for item_id, feature in zip(item_ids, features)}


def safe_image_feature_dict(ids_path: str, feat_path: str) -> Dict[str, np.ndarray]:
    if not (os.path.exists(ids_path) and os.path.exists(feat_path)):
        return {}
    return load_feature_dict(ids_path, feat_path)


def feature_artifacts_need_refresh(config: ExperimentConfig) -> tuple[bool, str]:
    item_corpus_path = Path(config.outputs_dir) / 'item_corpus.csv'
    required_paths = [
        config.data_csv_path,
        config.text_feat_ids_path_resolved,
        config.text_feat_path_resolved,
        config.image_feat_ids_path_resolved,
        config.image_feat_path_resolved,
        str(item_corpus_path),
    ]
    missing_paths = [path for path in required_paths if not os.path.exists(path)]
    if missing_paths:
        return True, f"missing artifacts: {', '.join(missing_paths)}"

    data_df = pd.read_csv(config.data_csv_path, usecols=['item_id'])
    data_item_ids = set(data_df['item_id'].astype(str))
    item_corpus_df = pd.read_csv(item_corpus_path, usecols=['item_id'])
    item_corpus_ids = set(item_corpus_df['item_id'].astype(str))
    text_ids = set(str(x) for x in np.load(config.text_feat_ids_path_resolved, allow_pickle=True))
    image_ids = set(str(x) for x in np.load(config.image_feat_ids_path_resolved, allow_pickle=True))

    if item_corpus_ids != data_item_ids:
        missing_count = len(data_item_ids - item_corpus_ids)
        extra_count = len(item_corpus_ids - data_item_ids)
        return True, (
            f"outputs/item_corpus.csv does not match {config.data_csv_path} "
            f"(missing={missing_count}, extra={extra_count})"
        )

    if text_ids != item_corpus_ids:
        missing_count = len(item_corpus_ids - text_ids)
        extra_count = len(text_ids - item_corpus_ids)
        return True, (
            "text features do not match outputs/item_corpus.csv "
            f"(missing={missing_count}, extra={extra_count})"
        )

    if image_ids != item_corpus_ids:
        missing_count = len(item_corpus_ids - image_ids)
        extra_count = len(image_ids - item_corpus_ids)
        return True, (
            "image features do not match outputs/item_corpus.csv "
            f"(missing={missing_count}, extra={extra_count})"
        )

    return False, "artifacts are aligned"


def leave_one_out_split(examples: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_parts = []
    test_parts = []

    for _, group in examples.groupby("user_id", sort=False):
        group = group.sort_values("time").reset_index(drop=True)
        if len(group) >= 2:
            train_parts.append(group.iloc[:-1])
            test_parts.append(group.iloc[-1:])
        else:
            train_parts.append(group)

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=examples.columns)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=examples.columns)
    return train_df, test_df


@dataclass
class RecommendationArtifacts:
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    user2idx: Dict[str, int]
    item_text_emb: Dict[str, np.ndarray]
    item_img_emb: Dict[str, np.ndarray]
    text_dim: int
    img_dim: int


def build_recommendation_artifacts(config: ExperimentConfig) -> RecommendationArtifacts:
    df = pd.read_csv(config.data_csv_path)
    interactions = prepare_interactions(df)

    item_text_emb = load_feature_dict(config.text_feat_ids_path_resolved, config.text_feat_path_resolved)
    item_img_emb = safe_image_feature_dict(config.image_feat_ids_path_resolved, config.image_feat_path_resolved)

    available_items = set(item_text_emb.keys())
    if item_img_emb:
        available_items &= set(item_img_emb.keys())

    item_text_emb = {item_id: item_text_emb[item_id] for item_id in available_items}
    if item_img_emb:
        item_img_emb = {item_id: item_img_emb[item_id] for item_id in available_items}

    interactions = interactions[interactions["item_id"].isin(available_items)].copy()

    examples = build_sequence_examples(
        interactions,
        max_seq_len=config.max_seq_len,
        min_history_len=config.min_history_len,
    )
    examples = examples[examples["item_id"].isin(available_items)].reset_index(drop=True)
    if examples.empty:
        raise RuntimeError("No valid sequence examples were built from the current data.")

    train_df, test_df = leave_one_out_split(examples)
    unique_users = train_df["user_id"].astype(str).unique().tolist()
    user2idx = {user_id: idx for idx, user_id in enumerate(unique_users)}

    train_df = train_df[train_df["user_id"].isin(user2idx)].reset_index(drop=True)
    test_df = test_df[test_df["user_id"].isin(user2idx)].reset_index(drop=True)

    text_dim = next(iter(item_text_emb.values())).shape[0]
    img_dim = next(iter(item_img_emb.values())).shape[0] if item_img_emb else 2048

    return RecommendationArtifacts(
        train_df=train_df,
        test_df=test_df,
        user2idx=user2idx,
        item_text_emb=item_text_emb,
        item_img_emb=item_img_emb,
        text_dim=text_dim,
        img_dim=img_dim,
    )


class SequenceDataset(Dataset):
    def __init__(
        self,
        examples: pd.DataFrame,
        user2idx: Dict[str, int],
        item_text_emb: Dict[str, np.ndarray],
        item_img_emb: Dict[str, np.ndarray],
        img_dim: int,
        max_seq_len: int,
    ):
        self.examples = examples.reset_index(drop=True)
        self.user2idx = user2idx
        self.item_text_emb = item_text_emb
        self.item_img_emb = item_img_emb
        self.text_dim = next(iter(item_text_emb.values())).shape[0]
        self.img_dim = img_dim
        self.max_seq_len = max_seq_len

    def __len__(self) -> int:
        return len(self.examples)

    def _get_text_vec(self, item_id: str) -> np.ndarray:
        return self.item_text_emb.get(item_id, np.zeros(self.text_dim, dtype=np.float32))

    def _get_img_vec(self, item_id: str) -> np.ndarray:
        return self.item_img_emb.get(item_id, np.zeros(self.img_dim, dtype=np.float32))

    def __getitem__(self, index: int):
        row = self.examples.iloc[index]
        user_idx = self.user2idx[str(row["user_id"])]
        target_item_id = str(row["item_id"])
        history_item_ids: List[str] = list(row["history_item_ids"])[-self.max_seq_len:]

        history_text = np.zeros((self.max_seq_len, self.text_dim), dtype=np.float32)
        history_img = np.zeros((self.max_seq_len, self.img_dim), dtype=np.float32)
        history_mask = np.zeros((self.max_seq_len,), dtype=np.float32)

        start_idx = self.max_seq_len - len(history_item_ids)
        for offset, item_id in enumerate(history_item_ids):
            history_text[start_idx + offset] = self._get_text_vec(item_id)
            history_img[start_idx + offset] = self._get_img_vec(item_id)
            history_mask[start_idx + offset] = 1.0

        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(self._get_text_vec(target_item_id), dtype=torch.float32),
            torch.tensor(self._get_img_vec(target_item_id), dtype=torch.float32),
            torch.tensor(history_text, dtype=torch.float32),
            torch.tensor(history_img, dtype=torch.float32),
            torch.tensor(history_mask, dtype=torch.float32),
            torch.tensor(float(row["rating"]), dtype=torch.float32),
        )


class PairwiseSequenceDataset(Dataset):
    def __init__(
        self,
        examples: pd.DataFrame,
        user2idx: Dict[str, int],
        item_text_emb: Dict[str, np.ndarray],
        item_img_emb: Dict[str, np.ndarray],
        img_dim: int,
        max_seq_len: int,
        all_item_ids: List[str],
        user_interacted_items: Dict[str, set[str]],
        positive_threshold: float,
        random_seed: int,
    ):
        self.examples = examples[examples["rating"] >= positive_threshold].reset_index(drop=True)
        self.user2idx = user2idx
        self.item_text_emb = item_text_emb
        self.item_img_emb = item_img_emb
        self.text_dim = next(iter(item_text_emb.values())).shape[0]
        self.img_dim = img_dim
        self.max_seq_len = max_seq_len
        self.all_item_ids = np.array([str(item_id) for item_id in all_item_ids], dtype=object)
        self.user_negative_candidates: Dict[str, np.ndarray] = {}
        self.rng = np.random.default_rng(random_seed)

        for user_id in self.examples["user_id"].astype(str).unique():
            interacted = user_interacted_items.get(user_id, set())
            candidates = [item_id for item_id in self.all_item_ids if item_id not in interacted]
            if not candidates:
                row_items = self.examples[self.examples["user_id"].astype(str) == user_id]["item_id"].astype(str).unique().tolist()
                blocked = set(row_items)
                candidates = [item_id for item_id in self.all_item_ids if item_id not in blocked]
            if not candidates:
                candidates = self.all_item_ids.tolist()
            self.user_negative_candidates[user_id] = np.array(candidates, dtype=object)

    def __len__(self) -> int:
        return len(self.examples)

    def _get_text_vec(self, item_id: str) -> np.ndarray:
        return self.item_text_emb.get(item_id, np.zeros(self.text_dim, dtype=np.float32))

    def _get_img_vec(self, item_id: str) -> np.ndarray:
        return self.item_img_emb.get(item_id, np.zeros(self.img_dim, dtype=np.float32))

    def __getitem__(self, index: int):
        row = self.examples.iloc[index]
        user_id = str(row["user_id"])
        user_idx = self.user2idx[user_id]
        pos_item_id = str(row["item_id"])
        history_item_ids: List[str] = list(row["history_item_ids"])[-self.max_seq_len:]

        candidate_items = self.user_negative_candidates[user_id]
        neg_item_id = str(candidate_items[self.rng.integers(len(candidate_items))])
        if neg_item_id == pos_item_id and len(candidate_items) > 1:
            neg_item_id = str(candidate_items[(self.rng.integers(len(candidate_items) - 1) + 1) % len(candidate_items)])

        history_text = np.zeros((self.max_seq_len, self.text_dim), dtype=np.float32)
        history_img = np.zeros((self.max_seq_len, self.img_dim), dtype=np.float32)
        history_mask = np.zeros((self.max_seq_len,), dtype=np.float32)
        start_idx = self.max_seq_len - len(history_item_ids)

        for offset, item_id in enumerate(history_item_ids):
            history_text[start_idx + offset] = self._get_text_vec(item_id)
            history_img[start_idx + offset] = self._get_img_vec(item_id)
            history_mask[start_idx + offset] = 1.0

        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(self._get_text_vec(pos_item_id), dtype=torch.float32),
            torch.tensor(self._get_img_vec(pos_item_id), dtype=torch.float32),
            torch.tensor(self._get_text_vec(neg_item_id), dtype=torch.float32),
            torch.tensor(self._get_img_vec(neg_item_id), dtype=torch.float32),
            torch.tensor(history_text, dtype=torch.float32),
            torch.tensor(history_img, dtype=torch.float32),
            torch.tensor(history_mask, dtype=torch.float32),
        )


class PairwisePointwiseDataset(Dataset):
    def __init__(
        self,
        examples: pd.DataFrame,
        user2idx: Dict[str, int],
        item_text_emb: Dict[str, np.ndarray],
        item_img_emb: Dict[str, np.ndarray],
        img_dim: int,
        all_item_ids: List[str],
        user_interacted_items: Dict[str, set[str]],
        positive_threshold: float,
        random_seed: int,
    ):
        self.examples = examples[examples["rating"] >= positive_threshold].reset_index(drop=True)
        self.user2idx = user2idx
        self.item_text_emb = item_text_emb
        self.item_img_emb = item_img_emb
        self.text_dim = next(iter(item_text_emb.values())).shape[0]
        self.img_dim = img_dim
        self.all_item_ids = np.array([str(item_id) for item_id in all_item_ids], dtype=object)
        self.user_negative_candidates: Dict[str, np.ndarray] = {}
        self.rng = np.random.default_rng(random_seed)

        for user_id in self.examples["user_id"].astype(str).unique():
            interacted = user_interacted_items.get(user_id, set())
            candidates = [item_id for item_id in self.all_item_ids if item_id not in interacted]
            if not candidates:
                row_items = self.examples[self.examples["user_id"].astype(str) == user_id]["item_id"].astype(str).unique().tolist()
                blocked = set(row_items)
                candidates = [item_id for item_id in self.all_item_ids if item_id not in blocked]
            if not candidates:
                candidates = self.all_item_ids.tolist()
            self.user_negative_candidates[user_id] = np.array(candidates, dtype=object)

    def __len__(self) -> int:
        return len(self.examples)

    def _get_text_vec(self, item_id: str) -> np.ndarray:
        return self.item_text_emb.get(item_id, np.zeros(self.text_dim, dtype=np.float32))

    def _get_img_vec(self, item_id: str) -> np.ndarray:
        return self.item_img_emb.get(item_id, np.zeros(self.img_dim, dtype=np.float32))

    def __getitem__(self, index: int):
        row = self.examples.iloc[index]
        user_id = str(row["user_id"])
        user_idx = self.user2idx[user_id]
        pos_item_id = str(row["item_id"])

        candidate_items = self.user_negative_candidates[user_id]
        neg_item_id = str(candidate_items[self.rng.integers(len(candidate_items))])
        if neg_item_id == pos_item_id and len(candidate_items) > 1:
            neg_item_id = str(candidate_items[(self.rng.integers(len(candidate_items) - 1) + 1) % len(candidate_items)])

        return (
            torch.tensor(user_idx, dtype=torch.long),
            torch.tensor(self._get_text_vec(pos_item_id), dtype=torch.float32),
            torch.tensor(self._get_img_vec(pos_item_id), dtype=torch.float32),
            torch.tensor(self._get_text_vec(neg_item_id), dtype=torch.float32),
            torch.tensor(self._get_img_vec(neg_item_id), dtype=torch.float32),
        )


class PointwiseDataset(Dataset):
    def __init__(
        self,
        examples: pd.DataFrame,
        user2idx: Dict[str, int],
        item_text_emb: Dict[str, np.ndarray],
        item_img_emb: Dict[str, np.ndarray],
        img_dim: int,
    ):
        self.examples = examples.reset_index(drop=True)
        self.user2idx = user2idx
        self.item_text_emb = item_text_emb
        self.item_img_emb = item_img_emb
        self.text_dim = next(iter(item_text_emb.values())).shape[0]
        self.img_dim = img_dim

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        row = self.examples.iloc[index]
        item_id = str(row["item_id"])
        return (
            torch.tensor(self.user2idx[str(row["user_id"])], dtype=torch.long),
            torch.tensor(self.item_text_emb.get(item_id, np.zeros(self.text_dim, dtype=np.float32)), dtype=torch.float32),
            torch.tensor(self.item_img_emb.get(item_id, np.zeros(self.img_dim, dtype=np.float32)), dtype=torch.float32),
            torch.tensor(float(row["rating"]), dtype=torch.float32),
        )
