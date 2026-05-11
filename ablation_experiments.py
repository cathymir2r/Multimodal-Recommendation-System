import copy
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
from recommendation_data import SequenceDataset, build_recommendation_artifacts, feature_artifacts_need_refresh


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def ensure_image_features(config: ExperimentConfig) -> None:
    refresh_needed, reason = feature_artifacts_need_refresh(config)
    if not refresh_needed:
        return
    print(f'Feature artifacts need refresh: {reason}')
    print(f'Extracting multimodal artifacts for backbone={config.image_backbone}...')
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


def evaluate_pointwise(model, loader, criterion, positive_threshold: float):
    model.eval()
    total_loss = 0.0
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

            total_loss += loss.item() * rating.size(0)
            total_mae += torch.abs(pred - rating).sum().item()
            total_alignment += aux['alignment_loss'].item() * rating.size(0)
            total_count += rating.size(0)
            y_true.extend((rating >= positive_threshold).long().cpu().numpy().tolist())
            y_pred.extend((pred >= positive_threshold).long().cpu().numpy().tolist())

    scale = max(total_count, 1)
    return {
        'mse': total_loss / scale,
        'mae': total_mae / scale,
        'alignment_loss': total_alignment / scale,
        **binary_classification_metrics(y_true, y_pred),
    }


def compute_topk_metrics(model, artifacts, config: ExperimentConfig, k_list=None):
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

            score_parts = []
            for start in range(0, text_feats.size(0), item_batch_size):
                end = start + item_batch_size
                batch_text = text_feats[start:end]
                batch_img = img_feats[start:end]
                batch_history_text = history_text.repeat(batch_text.size(0), 1, 1)
                batch_history_img = history_img.repeat(batch_text.size(0), 1, 1)
                batch_history_mask = history_mask.repeat(batch_text.size(0), 1)
                user_idx = torch.tensor([artifacts.user2idx[user_id]], dtype=torch.long, device=DEVICE).repeat(batch_text.size(0))
                batch_scores = model(user_idx, batch_text, batch_img, batch_history_text, batch_history_img, batch_history_mask)
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


def run_single_ablation(config: ExperimentConfig, artifacts, model_name: str):
    set_global_seed(config.random_seed)
    train_dataset = SequenceDataset(artifacts.train_df, artifacts.user2idx, artifacts.item_text_emb, artifacts.item_img_emb, artifacts.img_dim, config.max_seq_len)
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

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    best_mse = float('inf')
    history = []
    model_path = os.path.join(config.model_dir, f'{model_name}.pt')

    for epoch in range(1, config.num_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, config.alignment_weight)
        val_metrics = evaluate_pointwise(model, test_loader, criterion, config.positive_threshold)
        history.append({
            'epoch': epoch,
            'train_loss': train_metrics['loss'],
            'train_mse': train_metrics['mse'],
            'train_alignment_loss': train_metrics['alignment_loss'],
            **val_metrics,
        })

        if val_metrics['mse'] < best_mse:
            best_mse = val_metrics['mse']
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_name': model_name,
                'config': config.to_dict(),
            }, model_path)

    checkpoint = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    topk, ranking_evaluation = compute_topk_metrics(model, artifacts, config)
    return {
        'model_name': model_name,
        'config': config.to_dict(),
        'best_val_mse': best_mse,
        'ranking_evaluation': ranking_evaluation,
        'epochs': history,
        'topk': topk,
    }


def main():
    base_config = ExperimentConfig()
    set_global_seed(base_config.random_seed)
    base_config.ensure_dirs()

    ablation_settings = [
        ('ablation_gru_encoder', {'sequence_encoder_type': 'gru'}),
        ('ablation_transformer_encoder', {'sequence_encoder_type': 'transformer'}),
        ('ablation_no_alignment', {'alignment_weight': 0.0}),
        ('ablation_high_alignment', {'alignment_weight': 0.1}),
        ('ablation_resnet_backbone', {'image_backbone': 'resnet50'}),
        ('ablation_vit_backbone', {'image_backbone': 'vit_b_16'}),
    ]

    results = []
    for model_name, overrides in ablation_settings:
        config = copy.deepcopy(base_config)
        for key, value in overrides.items():
            setattr(config, key, value)
        config.ensure_dirs()
        ensure_image_features(config)
        artifacts = build_recommendation_artifacts(config)
        print(f'Running ablation: {model_name}')
        results.append(run_single_ablation(config, artifacts, model_name))

    report = {
        'base_config': base_config.to_dict(),
        'results': results,
    }
    save_json_report(os.path.join(base_config.reports_dir, 'ablation_report.json'), report)


if __name__ == '__main__':
    main()
