import hashlib
import json
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch


def binary_classification_metrics(y_true: List[int], y_pred: List[int]) -> dict:
    tp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 1)
    fp = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 0 and pred == 1)
    fn = sum(1 for truth, pred in zip(y_true, y_pred) if truth == 1 and pred == 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def save_json_report(report_path: str, payload: dict) -> None:
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def _update_topk_metrics(
    ranked_item_ids: np.ndarray,
    pos_items: set[str],
    k_list: List[int],
    metrics: Dict[int, Dict[str, float]],
    raw_counts: Dict[int, Dict[str, float]] | None = None,
) -> None:
    for k in k_list:
        topk_items = ranked_item_ids[:k]
        hit = any(item_id in pos_items for item_id in topk_items)
        if hit:
            metrics[k]["hit"] += 1.0
            if raw_counts is not None:
                raw_counts[k]["hit_users"] += 1.0

        dcg = 0.0
        for rank, item_id in enumerate(topk_items, start=1):
            if item_id in pos_items:
                dcg += 1.0 / np.log2(rank + 1)

        ideal_len = min(len(pos_items), k)
        idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_len + 1))
        ndcg = dcg / idcg if idcg > 0 else 0.0
        metrics[k]["ndcg"] += ndcg
        if raw_counts is not None:
            raw_counts[k]["ndcg_total"] += ndcg


def _candidate_count_summary(candidate_counts: List[int]) -> dict:
    if not candidate_counts:
        return {
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
        }

    counts = np.array(candidate_counts, dtype=np.int64)
    return {
        "min": int(counts.min()),
        "max": int(counts.max()),
        "mean": float(counts.mean()),
        "median": float(np.median(counts)),
    }


def compute_topk_from_scores(
    user_scores: Dict[str, np.ndarray],
    item_ids: np.ndarray,
    user_targets: Dict[str, set],
    k_list: List[int],
) -> Dict[int, Dict[str, float]]:
    metrics = {k: {"hit": 0.0, "ndcg": 0.0} for k in k_list}
    evaluated_users = 0

    for user_id, scores in user_scores.items():
        pos_items = user_targets.get(user_id)
        if not pos_items:
            continue

        ranked_item_ids = item_ids[np.argsort(-scores)]
        evaluated_users += 1

        _update_topk_metrics(ranked_item_ids, pos_items, k_list, metrics)

    if evaluated_users:
        for k in k_list:
            metrics[k]["hit"] /= evaluated_users
            metrics[k]["ndcg"] /= evaluated_users

    return metrics


def build_user_targets(df_test: pd.DataFrame, positive_threshold: float | None = None) -> Dict[str, set]:
    user_targets: Dict[str, set] = {}
    work_df = df_test
    if positive_threshold is not None and "rating" in work_df.columns:
        work_df = work_df[work_df["rating"] >= positive_threshold]

    for _, row in work_df.iterrows():
        user_id = str(row["user_id"])
        user_targets.setdefault(user_id, set()).add(str(row["item_id"]))
    return user_targets


def build_user_seen_items(df_test: pd.DataFrame) -> Dict[str, set]:
    user_seen: Dict[str, set] = {}
    for _, row in df_test.iterrows():
        user_id = str(row["user_id"])
        history_items = set(str(item_id) for item_id in row.get("history_item_ids", []))
        user_seen.setdefault(user_id, set()).update(history_items)
    return user_seen


def _stable_user_seed(user_id: str, random_seed: int) -> int:
    digest = hashlib.md5(user_id.encode("utf-8")).hexdigest()
    return (int(digest[:8], 16) + int(random_seed)) % (2**32)


def _random_hit_probability(num_candidates: int, num_positives: int, k: int) -> float:
    if num_candidates <= 0 or num_positives <= 0 or k <= 0:
        return 0.0
    topk = min(k, num_candidates)
    if num_positives >= num_candidates or topk >= num_candidates:
        return 1.0

    miss_prob = 1.0
    negatives = num_candidates - num_positives
    for draw_idx in range(topk):
        miss_prob *= max(negatives - draw_idx, 0) / (num_candidates - draw_idx)
    return 1.0 - miss_prob


def _random_ndcg_expectation(num_candidates: int, num_positives: int, k: int) -> float:
    if num_candidates <= 0 or num_positives <= 0 or k <= 0:
        return 0.0

    topk = min(k, num_candidates)
    ideal_len = min(num_positives, topk)
    idcg = sum(1.0 / np.log2(rank + 1) for rank in range(1, ideal_len + 1))
    if idcg <= 0:
        return 0.0

    expected_dcg = (num_positives / num_candidates) * sum(
        1.0 / np.log2(rank + 1) for rank in range(1, topk + 1)
    )
    return expected_dcg / idcg


def _random_baseline_summary(candidate_sizes: List[int], positive_sizes: List[int], k_list: List[int]) -> dict:
    if not candidate_sizes:
        return {str(k): {"hit": 0.0, "ndcg": 0.0, "expected_hit_users": 0.0} for k in k_list}

    summary = {}
    total_users = len(candidate_sizes)
    for k in k_list:
        hit_probs = [
            _random_hit_probability(num_candidates, num_positives, k)
            for num_candidates, num_positives in zip(candidate_sizes, positive_sizes)
        ]
        ndcg_expectations = [
            _random_ndcg_expectation(num_candidates, num_positives, k)
            for num_candidates, num_positives in zip(candidate_sizes, positive_sizes)
        ]
        mean_hit = float(np.mean(hit_probs))
        summary[str(k)] = {
            "hit": mean_hit,
            "ndcg": float(np.mean(ndcg_expectations)),
            "expected_hit_users": mean_hit * total_users,
        }
    return summary


def compute_ranking_metrics_from_scores(
    user_scores: Dict[str, np.ndarray],
    item_ids: np.ndarray,
    user_targets: Dict[str, set],
    user_seen_items: Dict[str, set],
    k_list: List[int],
    candidate_mode: str = "full",
    num_negatives: int = 99,
    random_seed: int = 42,
) -> Tuple[Dict[int, Dict[str, float]], dict]:
    if candidate_mode not in {"full", "sampled"}:
        raise ValueError(f"Unsupported ranking evaluation mode: {candidate_mode}")

    metrics = {k: {"hit": 0.0, "ndcg": 0.0} for k in k_list}
    raw_counts = {k: {"hit_users": 0.0, "ndcg_total": 0.0} for k in k_list}
    evaluated_users = 0
    candidate_counts: List[int] = []
    positive_counts: List[int] = []
    item_ids = np.array([str(item_id) for item_id in item_ids])
    all_items = item_ids.tolist()
    item_to_idx = {item_id: idx for idx, item_id in enumerate(all_items)}

    for user_id in sorted(user_scores.keys()):
        pos_items = {
            str(item_id)
            for item_id in user_targets.get(user_id, set())
            if str(item_id) in item_to_idx
        }
        if not pos_items:
            continue

        seen_items = {str(item_id) for item_id in user_seen_items.get(user_id, set())}
        candidate_negatives = [
            item_id for item_id in all_items
            if item_id not in pos_items and item_id not in seen_items
        ]

        if candidate_mode == "sampled":
            if not candidate_negatives:
                continue
            rng = np.random.default_rng(_stable_user_seed(user_id, random_seed))
            sample_size = min(num_negatives, len(candidate_negatives))
            sampled_negatives = rng.choice(candidate_negatives, size=sample_size, replace=False).tolist()
            candidate_items = list(pos_items) + sampled_negatives
        else:
            candidate_items = [
                item_id for item_id in all_items
                if item_id in pos_items or item_id not in seen_items
            ]
            if len(candidate_items) <= len(pos_items):
                continue

        candidate_indices = [item_to_idx[item_id] for item_id in candidate_items if item_id in item_to_idx]
        if not candidate_indices:
            continue

        candidate_scores = user_scores[user_id][candidate_indices]
        ranked_local_indices = np.argsort(-candidate_scores)
        ranked_item_ids = np.array(candidate_items, dtype=object)[ranked_local_indices]
        evaluated_users += 1
        candidate_counts.append(len(candidate_items))
        positive_counts.append(len(pos_items))
        _update_topk_metrics(ranked_item_ids, pos_items, k_list, metrics, raw_counts=raw_counts)

    if evaluated_users:
        for k in k_list:
            metrics[k]["hit"] /= evaluated_users
            metrics[k]["ndcg"] /= evaluated_users

    metadata = {
        "mode": candidate_mode,
        "num_negatives": int(num_negatives) if candidate_mode == "sampled" else None,
        "evaluated_users": int(evaluated_users),
        "candidate_count": _candidate_count_summary(candidate_counts),
        "random_baseline": _random_baseline_summary(candidate_counts, positive_counts, k_list),
        "observed_counts": {
            str(k): {
                "hit_users": int(raw_counts[k]["hit_users"]),
                "ndcg_total": float(raw_counts[k]["ndcg_total"]),
            }
            for k in k_list
        },
    }
    return metrics, metadata


def compute_sampled_topk_from_scores(
    user_scores: Dict[str, np.ndarray],
    item_ids: np.ndarray,
    user_targets: Dict[str, set],
    user_seen_items: Dict[str, set],
    k_list: List[int],
    num_negatives: int = 99,
    random_seed: int = 42,
) -> Dict[int, Dict[str, float]]:
    metrics, _ = compute_ranking_metrics_from_scores(
        user_scores,
        item_ids,
        user_targets,
        user_seen_items,
        k_list,
        candidate_mode="sampled",
        num_negatives=num_negatives,
        random_seed=random_seed,
    )
    return metrics


def compute_exact_random_baseline(
    item_ids: np.ndarray,
    user_targets: Dict[str, set],
    user_seen_items: Dict[str, set],
    k_list: List[int],
    candidate_mode: str = "full",
    num_negatives: int = 99,
    random_seed: int = 42,
) -> Tuple[Dict[int, Dict[str, float]], dict]:
    if candidate_mode not in {"full", "sampled"}:
        raise ValueError(f"Unsupported ranking evaluation mode: {candidate_mode}")

    item_ids = np.array([str(item_id) for item_id in item_ids])
    all_items = item_ids.tolist()
    candidate_counts: List[int] = []
    positive_counts: List[int] = []

    for user_id in sorted(user_targets.keys()):
        pos_items = {
            str(item_id)
            for item_id in user_targets.get(user_id, set())
            if str(item_id) in all_items
        }
        if not pos_items:
            continue

        seen_items = {str(item_id) for item_id in user_seen_items.get(user_id, set())}
        candidate_negatives = [
            item_id for item_id in all_items
            if item_id not in pos_items and item_id not in seen_items
        ]

        if candidate_mode == "sampled":
            if not candidate_negatives:
                continue
            rng = np.random.default_rng(_stable_user_seed(user_id, random_seed))
            sample_size = min(num_negatives, len(candidate_negatives))
            candidate_size = len(pos_items) + sample_size
        else:
            candidate_size = sum(1 for item_id in all_items if item_id in pos_items or item_id not in seen_items)
            if candidate_size <= len(pos_items):
                continue

        candidate_counts.append(candidate_size)
        positive_counts.append(len(pos_items))

    random_baseline = _random_baseline_summary(candidate_counts, positive_counts, k_list)
    metrics = {
        k: {
            "hit": random_baseline[str(k)]["hit"],
            "ndcg": random_baseline[str(k)]["ndcg"],
        }
        for k in k_list
    }
    metadata = {
        "mode": candidate_mode,
        "num_negatives": int(num_negatives) if candidate_mode == "sampled" else None,
        "evaluated_users": int(len(candidate_counts)),
        "candidate_count": _candidate_count_summary(candidate_counts),
        "random_baseline": random_baseline,
        "observed_counts": {
            str(k): {
                "hit_users": None,
                "ndcg_total": None,
            }
            for k in k_list
        },
    }
    return metrics, metadata


def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()
