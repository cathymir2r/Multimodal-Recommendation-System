import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json(path: str):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def save_json(path: str, payload: dict) -> None:
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)


def safe_metric(row: dict, key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _wrapped_labels(rows: List[dict]) -> List[str]:
    labels = []
    for row in rows:
        model_name = str(row.get('model_name', 'unknown'))
        labels.append(model_name.replace('_', '\n'))
    return labels


def _plot_bar_chart(rows: List[dict], metric_key: str, title: str, output_path: str, top_n: int = 8) -> str:
    usable_rows = [row for row in rows if row.get(metric_key) is not None]
    if not usable_rows:
        return ''
    usable_rows = sorted(usable_rows, key=lambda row: safe_metric(row, metric_key), reverse=True)[:top_n]
    labels = _wrapped_labels(usable_rows)
    values = [safe_metric(row, metric_key) for row in usable_rows]

    plt.figure(figsize=(10, 5))
    bars = plt.bar(labels, values, color='#0f766e')
    plt.title(title)
    plt.ylabel(metric_key)
    plt.xticks(rotation=0, fontsize=9)
    plt.ylim(0, max(values) * 1.15 if values else 1.0)
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, value, f'{value:.4f}', ha='center', va='bottom', fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
    return output_path


def _plot_ablation_chart(rows: List[dict], output_path: str) -> str:
    ablation_rows = [row for row in rows if row.get('group') == 'ablation']
    if not ablation_rows:
        return ''
    labels = _wrapped_labels(ablation_rows)
    mse_values = [safe_metric(row, 'best_val_mse') for row in ablation_rows]
    hit_values = [safe_metric(row, 'hit@10') for row in ablation_rows]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].bar(labels, mse_values, color='#b45309')
    axes[0].set_title('Ablation Validation MSE')
    axes[0].set_ylabel('best_val_mse')
    axes[0].tick_params(axis='x', labelrotation=0, labelsize=8)

    axes[1].bar(labels, hit_values, color='#1d4ed8')
    axes[1].set_title('Ablation Hit@10')
    axes[1].set_ylabel('hit@10')
    axes[1].tick_params(axis='x', labelrotation=0, labelsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _build_headline_insights(rows: List[dict]) -> List[str]:
    if not rows:
        return ['??????????????????????????']

    sorted_by_mse = sorted(rows, key=lambda row: safe_metric(row, 'best_val_mse', default=10**9))
    best_overall = sorted_by_mse[0]
    baseline_rows = [row for row in rows if row.get('group') == 'baseline']
    ablation_rows = [row for row in rows if row.get('group') == 'ablation']
    main_rows = [row for row in rows if row.get('group') == 'main']

    insights = [
        (
            f"???????????? {best_overall.get('model_name')}?"
            f"???? MSE ? {safe_metric(best_overall, 'best_val_mse'):.4f}?"
            f"F1 ? {safe_metric(best_overall, 'f1'):.4f}?Hit@10 ? {safe_metric(best_overall, 'hit@10'):.4f}?"
        )
    ]

    if baseline_rows:
        best_baseline = sorted(baseline_rows, key=lambda row: safe_metric(row, 'best_val_mse', default=10**9))[0]
        insights.append(
            f"?????????{best_baseline.get('model_name')} ??????"
            f"???? MSE ? {safe_metric(best_baseline, 'best_val_mse'):.4f}?"
        )

    if main_rows and baseline_rows:
        best_main = sorted(main_rows, key=lambda row: safe_metric(row, 'best_val_mse', default=10**9))[0]
        best_baseline = sorted(baseline_rows, key=lambda row: safe_metric(row, 'best_val_mse', default=10**9))[0]
        mse_gain = safe_metric(best_baseline, 'best_val_mse') - safe_metric(best_main, 'best_val_mse')
        hit_gain = safe_metric(best_main, 'hit@10') - safe_metric(best_baseline, 'hit@10')
        if mse_gain >= 0:
            insights.append(
                f"??????????????? MSE ??????? {mse_gain:.4f}?"
                f"?? Hit@10 ??? {hit_gain:.4f}?"
            )
        else:
            insights.append(
                f"????????? MSE ?????????? {abs(mse_gain):.4f}?"
                f"? Hit@10 ??? {hit_gain:.4f}??????????"
            )

    if ablation_rows:
        best_ablation = sorted(ablation_rows, key=lambda row: safe_metric(row, 'best_val_mse', default=10**9))[0]
        insights.append(
            f"???????{best_ablation.get('model_name')} ?????"
            f"???????? {best_ablation.get('image_backbone')}?"
            f"??????? {best_ablation.get('sequence_encoder_type')}?"
        )

    backbone_groups: Dict[str, List[dict]] = {}
    for row in rows:
        backbone = row.get('image_backbone')
        if backbone:
            backbone_groups.setdefault(backbone, []).append(row)
    if len(backbone_groups) >= 2:
        backbone_scores = {
            backbone: min(safe_metric(row, 'best_val_mse', default=10**9) for row in backbone_rows)
            for backbone, backbone_rows in backbone_groups.items()
        }
        ranked_backbones = sorted(backbone_scores.items(), key=lambda item: item[1])
        insights.append(
            f"??????????????{ranked_backbones[0][0]} ????????????? MSE?"
            f"??? {ranked_backbones[0][1]:.4f}?"
        )

    return insights


def _build_summary_cards(rows: List[dict]) -> List[dict]:
    if not rows:
        return []

    cards = []
    sorted_by_mse = sorted(rows, key=lambda row: safe_metric(row, 'best_val_mse', default=10**9))
    best_overall = sorted_by_mse[0]
    cards.append({
        'title': '????',
        'value': best_overall.get('model_name'),
        'detail': f"MSE {safe_metric(best_overall, 'best_val_mse'):.4f} | Hit@10 {safe_metric(best_overall, 'hit@10'):.4f}",
    })

    baseline_rows = [row for row in rows if row.get('group') == 'baseline']
    if baseline_rows:
        best_baseline = sorted(baseline_rows, key=lambda row: safe_metric(row, 'best_val_mse', default=10**9))[0]
        cards.append({
            'title': '????',
            'value': best_baseline.get('model_name'),
            'detail': f"MSE {safe_metric(best_baseline, 'best_val_mse'):.4f} | F1 {safe_metric(best_baseline, 'f1'):.4f}",
        })

    main_rows = [row for row in rows if row.get('group') == 'main']
    if main_rows and baseline_rows:
        best_main = sorted(main_rows, key=lambda row: safe_metric(row, 'best_val_mse', default=10**9))[0]
        best_baseline = sorted(baseline_rows, key=lambda row: safe_metric(row, 'best_val_mse', default=10**9))[0]
        cards.append({
            'title': '?????',
            'value': f"{safe_metric(best_baseline, 'best_val_mse') - safe_metric(best_main, 'best_val_mse'):.4f} MSE",
            'detail': (
                f"Hit@10 gain {safe_metric(best_main, 'hit@10') - safe_metric(best_baseline, 'hit@10'):.4f}"
            ),
        })

    backbone_groups: Dict[str, List[dict]] = {}
    for row in rows:
        backbone = row.get('image_backbone')
        if backbone:
            backbone_groups.setdefault(backbone, []).append(row)
    if backbone_groups:
        backbone_scores = {
            backbone: min(safe_metric(row, 'best_val_mse', default=10**9) for row in group_rows)
            for backbone, group_rows in backbone_groups.items()
        }
        best_backbone, best_score = sorted(backbone_scores.items(), key=lambda item: item[1])[0]
        cards.append({
            'title': '??????',
            'value': best_backbone,
            'detail': f"??????? MSE ? {best_score:.4f}",
        })

    return cards


def generate_experiment_assets(summary_rows: List[dict], reports_dir: str) -> dict:
    ensure_dir(reports_dir)
    figures_dir = os.path.join(reports_dir, 'figures')
    ensure_dir(figures_dir)

    charts = []
    chart_specs = [
        ('f1_overview.png', 'f1', 'Experiment F1 Comparison'),
        ('hit10_overview.png', 'hit@10', 'Experiment Hit@10 Comparison'),
        ('ndcg10_overview.png', 'ndcg@10', 'Experiment NDCG@10 Comparison'),
    ]
    for filename, metric_key, title in chart_specs:
        path = _plot_bar_chart(summary_rows, metric_key, title, os.path.join(figures_dir, filename))
        if path:
            charts.append({
                'name': filename,
                'metric': metric_key,
                'path': path,
                'caption': f'{title}. Higher values indicate stronger recommendation quality on {metric_key}.',
            })

    ablation_path = _plot_ablation_chart(summary_rows, os.path.join(figures_dir, 'ablation_overview.png'))
    if ablation_path:
        charts.append({
            'name': 'ablation_overview.png',
            'metric': 'ablation',
            'path': ablation_path,
            'caption': 'Ablation comparison across validation MSE and Hit@10 for different encoder and alignment settings.',
        })

    insights = _build_headline_insights(summary_rows)
    payload = {
        'highlights': insights,
        'summary_cards': _build_summary_cards(summary_rows),
        'charts': charts,
    }
    save_json(os.path.join(reports_dir, 'experiment_insights.json'), payload)
    return payload
