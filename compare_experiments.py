import csv
import json
import os

from experiment_config import ExperimentConfig
from experiment_utils import get_best_epoch
from experiment_reporting import generate_experiment_assets


def load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def rows_from_report(report, label=''):
    rows = []
    if not report:
        return rows
    if 'results' in report:
        iterable = report.get('results', [])
        for result in iterable:
            best_epoch = get_best_epoch(result.get('epochs', []))
            rows.append({
                'group': label,
                'model_name': result.get('model_name'),
                'image_backbone': result.get('config', {}).get('image_backbone'),
                'sequence_encoder_type': result.get('config', {}).get('sequence_encoder_type'),
                'alignment_weight': result.get('config', {}).get('alignment_weight'),
                'best_val_mse': result.get('best_val_mse'),
                'best_epoch': best_epoch.get('epoch'),
                'alignment_loss': best_epoch.get('alignment_loss') or best_epoch.get('train_alignment_loss'),
                'precision': best_epoch.get('precision'),
                'recall': best_epoch.get('recall'),
                'f1': best_epoch.get('f1'),
                'hit@5': result.get('topk', {}).get('5', {}).get('hit'),
                'ndcg@5': result.get('topk', {}).get('5', {}).get('ndcg'),
                'hit@10': result.get('topk', {}).get('10', {}).get('hit'),
                'ndcg@10': result.get('topk', {}).get('10', {}).get('ndcg'),
            })
    else:
        best_epoch = get_best_epoch(report.get('epochs', []))
        rows.append({
            'group': label,
            'model_name': report.get('model_name'),
            'image_backbone': report.get('config', {}).get('image_backbone'),
            'sequence_encoder_type': report.get('config', {}).get('sequence_encoder_type'),
            'alignment_weight': report.get('config', {}).get('alignment_weight'),
            'best_val_mse': report.get('best_val_mse'),
            'best_epoch': best_epoch.get('epoch'),
            'alignment_loss': best_epoch.get('alignment_loss') or best_epoch.get('train_alignment_loss'),
            'precision': best_epoch.get('precision'),
            'recall': best_epoch.get('recall'),
            'f1': best_epoch.get('f1'),
            'hit@5': report.get('topk', {}).get('5', {}).get('hit'),
            'ndcg@5': report.get('topk', {}).get('5', {}).get('ndcg'),
            'hit@10': report.get('topk', {}).get('10', {}).get('hit'),
            'ndcg@10': report.get('topk', {}).get('10', {}).get('ndcg'),
        })
    return rows


def main():
    config = ExperimentConfig()
    sequence_report = load_json(os.path.join(config.reports_dir, 'sequence_attention_report.json'))
    baseline_report = load_json(os.path.join(config.reports_dir, 'baseline_comparison_report.json'))
    ablation_report = load_json(os.path.join(config.reports_dir, 'ablation_report.json'))

    rows = []
    rows.extend(rows_from_report(baseline_report, 'baseline'))
    rows.extend(rows_from_report(sequence_report, 'main'))
    rows.extend(rows_from_report(ablation_report, 'ablation'))

    main_configs = {
        (
            row.get('image_backbone'),
            row.get('sequence_encoder_type'),
            row.get('alignment_weight'),
        )
        for row in rows
        if row.get('group') == 'main'
    }
    if main_configs:
        rows = [
            row
            for row in rows
            if row.get('group') != 'ablation'
            or (
                row.get('image_backbone'),
                row.get('sequence_encoder_type'),
                row.get('alignment_weight'),
            ) not in main_configs
        ]

    rows.sort(key=lambda row: (row['best_val_mse'] is None, row['best_val_mse']))
    os.makedirs(config.reports_dir, exist_ok=True)
    summary_json_path = os.path.join(config.reports_dir, 'experiment_summary.json')
    summary_csv_path = os.path.join(config.reports_dir, 'experiment_summary.csv')

    with open(summary_json_path, 'w', encoding='utf-8') as file:
        json.dump({'results': rows}, file, ensure_ascii=False, indent=2)

    if rows:
        with open(summary_csv_path, 'w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    insights = generate_experiment_assets(rows, config.reports_dir)

    print(f'saved experiment summary to {summary_json_path}')
    print(f'saved experiment summary to {summary_csv_path}')
    print(f"saved experiment insights to {os.path.join(config.reports_dir, 'experiment_insights.json')}")
    print(f"generated {len(insights.get('charts', []))} experiment figures")


if __name__ == '__main__':
    main()
