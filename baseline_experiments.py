import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from baseline_models import EarlyFusionRecommender, GatedFusionRecommender, ImageOnlyRecommender, TextOnlyRecommender
from evaluation import (
    binary_classification_metrics,
    build_user_seen_items,
    build_user_targets,
    compute_exact_random_baseline,
    compute_ranking_metrics_from_scores,
    save_json_report,
)
from experiment_config import ExperimentConfig
from recommendation_data import (
    PairwisePointwiseDataset,
    PointwiseDataset,
    build_recommendation_artifacts,
    feature_artifacts_need_refresh,
)
from experiment_utils import set_global_seed


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_required_features(config: ExperimentConfig) -> None:
    refresh_needed, reason = feature_artifacts_need_refresh(config)
    if not refresh_needed:
        return
    print(f"Feature artifacts need refresh: {reason}")
    print("Extracting required multimodal artifacts...")
    subprocess.run([sys.executable, "main.py", config.image_backbone, config.data_csv_path], check=True)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_count = 0

    for user_idx, text_vec, img_vec, rating in loader:
        user_idx = user_idx.to(DEVICE)
        text_vec = text_vec.to(DEVICE)
        img_vec = img_vec.to(DEVICE)
        rating = rating.to(DEVICE)

        optimizer.zero_grad()
        pred = model(user_idx, text_vec, img_vec)
        loss = criterion(pred, rating)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * rating.size(0)
        total_count += rating.size(0)

    return total_loss / max(total_count, 1)


def train_one_epoch_bpr(model, loader, optimizer):
    model.train()
    total_loss = 0.0
    total_count = 0

    for user_idx, pos_text, pos_img, neg_text, neg_img in loader:
        user_idx = user_idx.to(DEVICE)
        pos_text = pos_text.to(DEVICE)
        pos_img = pos_img.to(DEVICE)
        neg_text = neg_text.to(DEVICE)
        neg_img = neg_img.to(DEVICE)

        optimizer.zero_grad()
        pos_score = model(user_idx, pos_text, pos_img)
        neg_score = model(user_idx, neg_text, neg_img)
        loss = torch.nn.functional.softplus(-(pos_score - neg_score)).mean()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * user_idx.size(0)
        total_count += user_idx.size(0)

    return total_loss / max(total_count, 1)


def evaluate_pointwise(model, loader, criterion, positive_threshold):
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_count = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for user_idx, text_vec, img_vec, rating in loader:
            user_idx = user_idx.to(DEVICE)
            text_vec = text_vec.to(DEVICE)
            img_vec = img_vec.to(DEVICE)
            rating = rating.to(DEVICE)

            pred = model(user_idx, text_vec, img_vec)
            loss = criterion(pred, rating)

            total_loss += loss.item() * rating.size(0)
            total_mae += torch.abs(pred - rating).sum().item()
            total_count += rating.size(0)
            y_true.extend((rating >= positive_threshold).long().cpu().numpy().tolist())
            y_pred.extend((pred >= positive_threshold).long().cpu().numpy().tolist())

    return {
        "mse": total_loss / max(total_count, 1),
        "mae": total_mae / max(total_count, 1),
        **binary_classification_metrics(y_true, y_pred),
    }


def compute_pointwise_topk_metrics(model, artifacts, config: ExperimentConfig, k_list=None):
    k_list = k_list or [5, 10]
    item_ids = np.array(sorted(artifacts.item_text_emb.keys()))
    text_feats = torch.tensor(
        np.stack([artifacts.item_text_emb[item_id] for item_id in item_ids]),
        dtype=torch.float32,
        device=DEVICE,
    )
    if artifacts.item_img_emb:
        img_feats_np = np.stack([
            artifacts.item_img_emb.get(item_id, np.zeros(artifacts.img_dim, dtype=np.float32))
            for item_id in item_ids
        ])
    else:
        img_feats_np = np.zeros((len(item_ids), artifacts.img_dim), dtype=np.float32)
    img_feats = torch.tensor(img_feats_np, dtype=torch.float32, device=DEVICE)

    user_targets = build_user_targets(artifacts.test_df, positive_threshold=config.positive_threshold)
    user_seen_items = build_user_seen_items(artifacts.test_df)
    user_scores = {}
    model.eval()
    item_batch_size = config.ranking_item_batch_size
    with torch.no_grad():
        for user_id in user_targets:
            if user_id not in artifacts.user2idx:
                continue
            score_parts = []
            for start in range(0, len(item_ids), item_batch_size):
                end = start + item_batch_size
                batch_text = text_feats[start:end]
                batch_img = img_feats[start:end]
                user_idx = torch.tensor([artifacts.user2idx[user_id]], dtype=torch.long, device=DEVICE).repeat(batch_text.size(0))
                score_parts.append(model(user_idx, batch_text, batch_img).cpu())
            user_scores[user_id] = torch.cat(score_parts).numpy()

    return compute_ranking_metrics_from_scores(
        user_scores,
        item_ids,
        user_targets,
        user_seen_items,
        k_list,
        candidate_mode=config.ranking_eval_mode,
        num_negatives=config.ranking_num_negatives,
        random_seed=config.random_seed,
    )


def compute_popularity_topk_metrics(artifacts, config: ExperimentConfig, k_list=None):
    k_list = k_list or [5, 10]
    item_ids = np.array(sorted(artifacts.item_text_emb.keys()))
    popularity = artifacts.train_df["item_id"].astype(str).value_counts()
    # Use a tiny deterministic offset so ties break consistently by sorted item order.
    tie_break = np.linspace(1e-9, 0.0, num=len(item_ids), endpoint=False, dtype=np.float32)
    base_scores = np.array([float(popularity.get(item_id, 0.0)) for item_id in item_ids], dtype=np.float32) + tie_break

    user_targets = build_user_targets(artifacts.test_df, positive_threshold=config.positive_threshold)
    user_seen_items = build_user_seen_items(artifacts.test_df)
    user_scores = {
        user_id: base_scores.copy()
        for user_id in user_targets
        if user_id in artifacts.user2idx
    }

    return compute_ranking_metrics_from_scores(
        user_scores,
        item_ids,
        user_targets,
        user_seen_items,
        k_list,
        candidate_mode=config.ranking_eval_mode,
        num_negatives=config.ranking_num_negatives,
        random_seed=config.random_seed,
    )


def compute_exact_random_baseline_result(artifacts, config: ExperimentConfig, k_list=None):
    k_list = k_list or [5, 10]
    item_ids = np.array(sorted(artifacts.item_text_emb.keys()))
    user_targets = build_user_targets(artifacts.test_df, positive_threshold=config.positive_threshold)
    user_seen_items = build_user_seen_items(artifacts.test_df)
    topk, ranking_evaluation = compute_exact_random_baseline(
        item_ids,
        user_targets,
        user_seen_items,
        k_list,
        candidate_mode=config.ranking_eval_mode,
        num_negatives=config.ranking_num_negatives,
        random_seed=config.random_seed,
    )
    return {
        "model_name": "exact_random_ranker",
        "best_val_mse": None,
        "ranking_evaluation": ranking_evaluation,
        "epochs": [],
        "topk": topk,
    }


def compute_popularity_baseline_result(artifacts, config: ExperimentConfig, k_list=None):
    topk, ranking_evaluation = compute_popularity_topk_metrics(artifacts, config, k_list=k_list)
    return {
        "model_name": "popularity_ranker",
        "best_val_mse": None,
        "ranking_evaluation": ranking_evaluation,
        "epochs": [],
        "topk": topk,
    }


def run_single_experiment(model_name, model_factory, artifacts, config):
    set_global_seed(config.random_seed)
    training_objective = config.training_objective.lower()
    all_item_ids = sorted(artifacts.item_text_emb.keys())
    user_interacted_items = {}
    for df in (artifacts.train_df, artifacts.test_df):
        for _, row in df.iterrows():
            user_id = str(row["user_id"])
            user_interacted_items.setdefault(user_id, set()).add(str(row["item_id"]))
            user_interacted_items[user_id].update(str(item_id) for item_id in row["history_item_ids"])

    if training_objective == "bpr":
        train_dataset = PairwisePointwiseDataset(
            artifacts.train_df,
            artifacts.user2idx,
            artifacts.item_text_emb,
            artifacts.item_img_emb,
            artifacts.img_dim,
            all_item_ids=all_item_ids,
            user_interacted_items=user_interacted_items,
            positive_threshold=config.positive_threshold,
            random_seed=config.random_seed,
        )
    else:
        train_dataset = PointwiseDataset(
            artifacts.train_df,
            artifacts.user2idx,
            artifacts.item_text_emb,
            artifacts.item_img_emb,
            artifacts.img_dim,
        )
    test_dataset = PointwiseDataset(
        artifacts.test_df,
        artifacts.user2idx,
        artifacts.item_text_emb,
        artifacts.item_img_emb,
        artifacts.img_dim,
    )
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = model_factory().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    selection_metric = config.selection_metric.lower()
    if training_objective == "bpr" and selection_metric == "mse":
        selection_metric = "ndcg10"

    if selection_metric == "ndcg10":
        best_selection_value = float("-inf")
    else:
        best_selection_value = float("inf")
    best_mse = float("inf")
    best_epoch = None
    history = []
    model_path = os.path.join(config.model_dir, f"{model_name}.pt")

    for epoch in range(1, config.num_epochs + 1):
        if training_objective == "bpr":
            train_loss = train_one_epoch_bpr(model, train_loader, optimizer)
        else:
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_metrics = evaluate_pointwise(model, test_loader, criterion, config.positive_threshold)
        epoch_summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_mse": None if training_objective == "bpr" else train_loss,
            **val_metrics,
        }
        if selection_metric == "ndcg10":
            epoch_topk, _ = compute_pointwise_topk_metrics(model, artifacts, config, k_list=[10])
            epoch_summary["hit@10"] = epoch_topk[10]["hit"]
            epoch_summary["ndcg@10"] = epoch_topk[10]["ndcg"]

        history.append(epoch_summary)
        best_mse = min(best_mse, val_metrics["mse"])

        if selection_metric == "ndcg10":
            selection_value = epoch_summary["ndcg@10"]
            if epoch < config.selection_min_epoch:
                is_better = False
            else:
                is_better = (
                    selection_value > best_selection_value
                    or (
                        selection_value == best_selection_value
                        and best_epoch is not None
                        and val_metrics["mse"] < history[best_epoch - 1]["mse"]
                    )
                )
        else:
            selection_value = val_metrics["mse"]
            is_better = selection_value < best_selection_value

        if is_better:
            best_selection_value = selection_value
            best_epoch = epoch
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_name": model_name,
                "config": config.to_dict(),
                "best_epoch": epoch,
                "selection_metric": selection_metric,
                "selection_value": selection_value,
            }, model_path)

        if training_objective == "bpr":
            print(
                f"[{model_name}][Epoch {epoch}] train_rank={train_loss:.4f} "
                f"val_mse={val_metrics['mse']:.4f} val_mae={val_metrics['mae']:.4f} "
                f"precision={val_metrics['precision']:.4f} recall={val_metrics['recall']:.4f} f1={val_metrics['f1']:.4f}"
            )
        else:
            print(
                f"[{model_name}][Epoch {epoch}] train_mse={train_loss:.4f} "
                f"val_mse={val_metrics['mse']:.4f} val_mae={val_metrics['mae']:.4f} "
                f"precision={val_metrics['precision']:.4f} recall={val_metrics['recall']:.4f} f1={val_metrics['f1']:.4f}"
            )
        if selection_metric == "ndcg10":
            print(
                f"                         val_hit10={epoch_summary['hit@10']:.4f} "
                f"val_ndcg10={epoch_summary['ndcg@10']:.4f}"
            )

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    topk, ranking_evaluation = compute_pointwise_topk_metrics(model, artifacts, config)

    return {
        "model_name": model_name,
        "best_val_mse": best_mse,
        "best_epoch": best_epoch,
        "selection_metric": selection_metric,
        "selection_value": best_selection_value,
        "ranking_evaluation": ranking_evaluation,
        "epochs": history,
        "topk": topk,
    }


def main():
    config = ExperimentConfig()
    set_global_seed(config.random_seed)
    config.ensure_dirs()
    ensure_required_features(config)
    artifacts = build_recommendation_artifacts(config)

    experiments = {
        "text_only_mlp": lambda: TextOnlyRecommender(
            num_users=len(artifacts.user2idx),
            text_dim=artifacts.text_dim,
            user_emb_dim=config.user_emb_dim,
            hidden_dim=config.hidden_dim,
        ),
        "image_only_mlp": lambda: ImageOnlyRecommender(
            num_users=len(artifacts.user2idx),
            img_dim=artifacts.img_dim,
            user_emb_dim=config.user_emb_dim,
            hidden_dim=config.hidden_dim,
        ),
        "early_fusion_mlp": lambda: EarlyFusionRecommender(
            num_users=len(artifacts.user2idx),
            text_dim=artifacts.text_dim,
            img_dim=artifacts.img_dim,
            user_emb_dim=config.user_emb_dim,
            hidden_dim=config.hidden_dim,
        ),
        "gated_fusion_mlp": lambda: GatedFusionRecommender(
            num_users=len(artifacts.user2idx),
            text_dim=artifacts.text_dim,
            img_dim=artifacts.img_dim,
            user_emb_dim=config.user_emb_dim,
            hidden_dim=config.hidden_dim,
        ),
    }

    results = []
    for model_name, model_factory in experiments.items():
        print(f"Running baseline: {model_name}")
        results.append(run_single_experiment(model_name, model_factory, artifacts, config))
    results.append(compute_popularity_baseline_result(artifacts, config))
    results.append(compute_exact_random_baseline_result(artifacts, config))

    report = {
        "config": config.to_dict(),
        "results": results,
    }
    save_json_report(os.path.join(config.reports_dir, "baseline_comparison_report.json"), report)


if __name__ == "__main__":
    main()
