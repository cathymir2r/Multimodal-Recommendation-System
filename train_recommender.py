import os
import subprocess
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation import (
    binary_classification_metrics,
    build_user_seen_items,
    build_user_targets,
    compute_ranking_metrics_from_scores,
    save_json_report,
)
from experiment_config import ExperimentConfig
from experiment_utils import set_global_seed
from multimodal_models import SequenceAttentionRecommender
from recommendation_data import (
    PairwiseSequenceDataset,
    SequenceDataset,
    build_recommendation_artifacts,
    feature_artifacts_need_refresh,
)


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def build_regression_criterion(config: ExperimentConfig):
    loss_type = config.loss_type.lower()
    if loss_type in {'smooth_l1', 'huber'}:
        return nn.SmoothL1Loss(beta=1.0)
    return nn.MSELoss()


def ensure_required_features(config: ExperimentConfig) -> None:
    refresh_needed, reason = feature_artifacts_need_refresh(config)
    if not refresh_needed:
        return
    print(f'Feature artifacts need refresh: {reason}')
    print('Extracting required multimodal artifacts...')
    subprocess.run([sys.executable, 'main.py', config.image_backbone, config.data_csv_path], check=True)


def train_one_epoch(model, loader, criterion, optimizer, alignment_weight: float):
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_alignment = 0.0
    total_count = 0

    for batch in loader:
        user_idx, target_text, target_img, history_text, history_img, history_mask, rating = [x.to(DEVICE) for x in batch]
        optimizer.zero_grad()
        pred, aux = model(user_idx, target_text, target_img, history_text, history_img, history_mask, return_aux=True)
        mse_loss = criterion(pred, rating)
        alignment_loss = aux['alignment_loss']
        loss = mse_loss + alignment_weight * alignment_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * rating.size(0)
        total_mse += mse_loss.item() * rating.size(0)
        total_alignment += alignment_loss.item() * rating.size(0)
        total_count += rating.size(0)

    scale = max(total_count, 1)
    return {
        'loss': total_loss / scale,
        'mse': total_mse / scale,
        'alignment_loss': total_alignment / scale,
    }


def train_one_epoch_bpr(model, loader, optimizer, alignment_weight: float):
    model.train()
    total_loss = 0.0
    total_ranking = 0.0
    total_alignment = 0.0
    total_count = 0

    for batch in loader:
        user_idx, pos_text, pos_img, neg_text, neg_img, history_text, history_img, history_mask = [x.to(DEVICE) for x in batch]
        optimizer.zero_grad()

        pos_score, aux = model(user_idx, pos_text, pos_img, history_text, history_img, history_mask, return_aux=True)
        neg_score = model(user_idx, neg_text, neg_img, history_text, history_img, history_mask, return_aux=False)

        ranking_loss = torch.nn.functional.softplus(-(pos_score - neg_score)).mean()
        alignment_loss = aux['alignment_loss']
        loss = ranking_loss + alignment_weight * alignment_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * user_idx.size(0)
        total_ranking += ranking_loss.item() * user_idx.size(0)
        total_alignment += alignment_loss.item() * user_idx.size(0)
        total_count += user_idx.size(0)

    scale = max(total_count, 1)
    return {
        'loss': total_loss / scale,
        'ranking_loss': total_ranking / scale,
        'alignment_loss': total_alignment / scale,
    }


def evaluate_pointwise(model, loader, criterion, positive_threshold: float):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    total_alignment = 0.0
    total_count = 0
    y_true = []
    y_pred = []

    with torch.no_grad():
        for batch in loader:
            user_idx, target_text, target_img, history_text, history_img, history_mask, rating = [x.to(DEVICE) for x in batch]
            pred, aux = model(user_idx, target_text, target_img, history_text, history_img, history_mask, return_aux=True)
            loss = criterion(pred, rating)
            mse = torch.mean((pred - rating) ** 2)

            total_loss += loss.item() * rating.size(0)
            total_mse += mse.item() * rating.size(0)
            total_mae += torch.abs(pred - rating).sum().item()
            total_alignment += aux['alignment_loss'].item() * rating.size(0)
            total_count += rating.size(0)
            y_true.extend((rating >= positive_threshold).long().cpu().numpy().tolist())
            y_pred.extend((pred >= positive_threshold).long().cpu().numpy().tolist())

    scale = max(total_count, 1)
    return {
        'loss': total_loss / scale,
        'mse': total_mse / scale,
        'mae': total_mae / scale,
        'alignment_loss': total_alignment / scale,
        **binary_classification_metrics(y_true, y_pred),
    }


def compute_sequence_topk_metrics(model, artifacts, config: ExperimentConfig, k_list=None):
    k_list = k_list or [5, 10]
    model.eval()
    item_batch_size = config.ranking_item_batch_size

    item_ids = np.array(sorted(artifacts.item_text_emb.keys()))
    text_feats = torch.tensor(np.stack([artifacts.item_text_emb[item_id] for item_id in item_ids]), dtype=torch.float32, device=DEVICE)
    if artifacts.item_img_emb:
        img_feats_np = np.stack([artifacts.item_img_emb.get(item_id, np.zeros(artifacts.img_dim, dtype=np.float32)) for item_id in item_ids])
    else:
        img_feats_np = np.zeros((len(item_ids), artifacts.img_dim), dtype=np.float32)
    img_feats = torch.tensor(img_feats_np, dtype=torch.float32, device=DEVICE)

    user_histories = {}
    for _, row in artifacts.test_df.iterrows():
        user_histories.setdefault(str(row['user_id']), list(row['history_item_ids']))

    user_targets = build_user_targets(artifacts.test_df, positive_threshold=config.positive_threshold)
    user_seen_items = build_user_seen_items(artifacts.test_df)
    user_scores = {}
    with torch.no_grad():
        for user_id in user_targets:
            if user_id not in artifacts.user2idx:
                continue
            history_ids = user_histories.get(user_id, [])[-config.max_seq_len:]
            history_text_np = np.zeros((1, config.max_seq_len, artifacts.text_dim), dtype=np.float32)
            history_img_np = np.zeros((1, config.max_seq_len, artifacts.img_dim), dtype=np.float32)
            history_mask_np = np.zeros((1, config.max_seq_len), dtype=np.float32)
            start_idx = config.max_seq_len - len(history_ids)
            for offset, item_id in enumerate(history_ids):
                history_text_np[0, start_idx + offset] = artifacts.item_text_emb.get(item_id, np.zeros(artifacts.text_dim, dtype=np.float32))
                history_img_np[0, start_idx + offset] = artifacts.item_img_emb.get(item_id, np.zeros(artifacts.img_dim, dtype=np.float32))
                history_mask_np[0, start_idx + offset] = 1.0

            history_text = torch.tensor(history_text_np, dtype=torch.float32, device=DEVICE)
            history_img = torch.tensor(history_img_np, dtype=torch.float32, device=DEVICE)
            history_mask = torch.tensor(history_mask_np, dtype=torch.float32, device=DEVICE)
            history_state = model.encode_history_state(history_text, history_img, history_mask)

            score_parts = []
            for start in range(0, text_feats.size(0), item_batch_size):
                end = start + item_batch_size
                batch_text = text_feats[start:end]
                batch_img = img_feats[start:end]
                user_idx = torch.tensor([artifacts.user2idx[user_id]], dtype=torch.long, device=DEVICE).repeat(batch_text.size(0))
                batch_scores = model.score_with_history_state(user_idx, batch_text, batch_img, history_state)
                score_parts.append(batch_scores.cpu())

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


def main():
    config = ExperimentConfig()
    set_global_seed(config.random_seed)
    config.ensure_dirs()
    ensure_required_features(config)
    artifacts = build_recommendation_artifacts(config)

    training_objective = config.training_objective.lower()
    all_item_ids = sorted(artifacts.item_text_emb.keys())
    user_interacted_items = {}
    for df in (artifacts.train_df, artifacts.test_df):
        for _, row in df.iterrows():
            user_id = str(row['user_id'])
            user_interacted_items.setdefault(user_id, set()).add(str(row['item_id']))
            user_interacted_items[user_id].update(str(item_id) for item_id in row['history_item_ids'])

    if training_objective == 'bpr':
        train_dataset = PairwiseSequenceDataset(
            artifacts.train_df,
            artifacts.user2idx,
            artifacts.item_text_emb,
            artifacts.item_img_emb,
            artifacts.img_dim,
            config.max_seq_len,
            all_item_ids=all_item_ids,
            user_interacted_items=user_interacted_items,
            positive_threshold=config.positive_threshold,
            random_seed=config.random_seed,
        )
    else:
        train_dataset = SequenceDataset(
            artifacts.train_df,
            artifacts.user2idx,
            artifacts.item_text_emb,
            artifacts.item_img_emb,
            artifacts.img_dim,
            config.max_seq_len,
        )
    test_dataset = SequenceDataset(artifacts.test_df, artifacts.user2idx, artifacts.item_text_emb, artifacts.item_img_emb, artifacts.img_dim, config.max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    model = SequenceAttentionRecommender(
        num_users=len(artifacts.user2idx),
        text_dim=artifacts.text_dim,
        img_dim=artifacts.img_dim,
        hidden_dim=config.hidden_dim,
        user_emb_dim=config.user_emb_dim,
        dropout=config.dropout,
        sequence_encoder_type=config.sequence_encoder_type,
        num_attention_heads=config.num_attention_heads,
        num_transformer_layers=config.num_transformer_layers,
        max_seq_len=config.max_seq_len,
    ).to(DEVICE)

    criterion = build_regression_criterion(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    selection_metric = config.selection_metric.lower()
    if selection_metric == 'ndcg10':
        best_selection_value = float('-inf')
    else:
        best_selection_value = float('inf')
    best_epoch = None
    history = []
    model_path = os.path.join(config.model_dir, 'sequence_attention_recommender.pt')
    report_path = os.path.join(config.reports_dir, 'sequence_attention_report.json')

    for epoch in range(1, config.num_epochs + 1):
        if training_objective == 'bpr':
            train_metrics = train_one_epoch_bpr(model, train_loader, optimizer, config.alignment_weight)
        else:
            train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, config.alignment_weight)
        val_metrics = evaluate_pointwise(model, test_loader, criterion, config.positive_threshold)
        epoch_summary = {
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_alignment_loss': train_metrics['alignment_loss'],
            **val_metrics,
        }
        if training_objective == 'bpr':
            epoch_summary['train_ranking_loss'] = train_metrics['ranking_loss']
            epoch_summary['train_mse'] = None
        else:
            epoch_summary['train_mse'] = train_metrics['mse']

        if selection_metric == 'ndcg10':
            epoch_topk, _ = compute_sequence_topk_metrics(model, artifacts, config, k_list=[10])
            epoch_summary['hit@10'] = epoch_topk[10]['hit']
            epoch_summary['ndcg@10'] = epoch_topk[10]['ndcg']

        history.append(epoch_summary)

        if training_objective == 'bpr':
            print(
                f"[Epoch {epoch}] train_loss={train_metrics['loss']:.4f} train_rank={train_metrics['ranking_loss']:.4f} "
                f"train_align={train_metrics['alignment_loss']:.4f} val_loss={val_metrics['loss']:.4f} "
                f"val_mse={val_metrics['mse']:.4f} val_mae={val_metrics['mae']:.4f} val_align={val_metrics['alignment_loss']:.4f} "
                f"precision={val_metrics['precision']:.4f} recall={val_metrics['recall']:.4f} f1={val_metrics['f1']:.4f}"
            )
        else:
            print(
                f"[Epoch {epoch}] train_loss={train_metrics['loss']:.4f} train_mse={train_metrics['mse']:.4f} "
                f"train_align={train_metrics['alignment_loss']:.4f} val_loss={val_metrics['loss']:.4f} "
                f"val_mse={val_metrics['mse']:.4f} val_mae={val_metrics['mae']:.4f} val_align={val_metrics['alignment_loss']:.4f} "
                f"precision={val_metrics['precision']:.4f} recall={val_metrics['recall']:.4f} f1={val_metrics['f1']:.4f}"
            )
        if selection_metric == 'ndcg10':
            print(f"           val_hit10={epoch_summary['hit@10']:.4f} val_ndcg10={epoch_summary['ndcg@10']:.4f}")

        if selection_metric == 'ndcg10':
            selection_value = epoch_summary['ndcg@10']
            if epoch < config.selection_min_epoch:
                is_better = False
            else:
                is_better = (
                    selection_value > best_selection_value
                    or (selection_value == best_selection_value and best_epoch is not None and val_metrics['mse'] < history[best_epoch - 1]['mse'])
                )
        else:
            selection_value = val_metrics['mse']
            is_better = selection_value < best_selection_value

        if is_better:
            best_selection_value = selection_value
            best_epoch = epoch
            torch.save({
                'model_state_dict': model.state_dict(),
                'user2idx': artifacts.user2idx,
                'text_dim': artifacts.text_dim,
                'img_dim': artifacts.img_dim,
                'hidden_dim': config.hidden_dim,
                'user_emb_dim': config.user_emb_dim,
                'max_seq_len': config.max_seq_len,
                'sequence_encoder_type': config.sequence_encoder_type,
                'num_attention_heads': config.num_attention_heads,
                'num_transformer_layers': config.num_transformer_layers,
                'dropout': config.dropout,
                'config': config.to_dict(),
                'best_epoch': epoch,
                'selection_metric': selection_metric,
                'selection_value': selection_value,
            }, model_path)

    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    topk_metrics, ranking_evaluation = compute_sequence_topk_metrics(model, artifacts, config)

    report = {
        'model_name': 'sequence_attention_recommender',
        'config': config.to_dict(),
        'best_val_mse': min(epoch['mse'] for epoch in history),
        'best_epoch': best_epoch,
        'selection_metric': selection_metric,
        'selection_value': best_selection_value,
        'ranking_evaluation': ranking_evaluation,
        'epochs': history,
        'topk': topk_metrics,
    }
    save_json_report(report_path, report)

    print('Top-K metrics:')
    for k, metric in topk_metrics.items():
        print(f"Hit@{k}={metric['hit']:.4f}, NDCG@{k}={metric['ndcg']:.4f}")
    print(
        'Ranking evaluation: '
        f"mode={ranking_evaluation['mode']} evaluated_users={ranking_evaluation['evaluated_users']} "
        f"candidate_mean={ranking_evaluation['candidate_count']['mean']:.2f}"
    )


if __name__ == '__main__':
    main()
