import argparse
from pathlib import Path

import pandas as pd


DEFAULT_SOURCE = Path('outputs/subsets/amazon_clothing_top500_users_ge3.csv')
DEFAULT_OUTPUT = Path('outputs/subsets/amazon_clothing_seq_clean_u5_i3.csv')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Clean a recommendation subset for sequential modeling.')
    parser.add_argument('--source', type=Path, default=DEFAULT_SOURCE, help='Source CSV path.')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='Output CSV path.')
    parser.add_argument('--min-user-interactions', type=int, default=5, help='Minimum interactions per user after cleaning.')
    parser.add_argument('--min-item-interactions', type=int, default=3, help='Minimum interactions per item after cleaning.')
    parser.add_argument(
        '--dedupe-key',
        choices=['row', 'user_item', 'user_item_time', 'user_item_time_rating'],
        default='user_item_time',
        help='How to deduplicate interactions before iterative filtering.',
    )
    return parser.parse_args()


def dedupe_dataframe(df: pd.DataFrame, dedupe_key: str) -> pd.DataFrame:
    if dedupe_key == 'row':
        return df.drop_duplicates().copy()
    if dedupe_key == 'user_item':
        subset = ['user_id', 'item_id']
    elif dedupe_key == 'user_item_time_rating':
        subset = ['user_id', 'item_id', 'time', 'rating']
    else:
        subset = ['user_id', 'item_id', 'time']
    return df.drop_duplicates(subset=subset, keep='last').copy()


def iterative_filter(df: pd.DataFrame, min_user_interactions: int, min_item_interactions: int) -> pd.DataFrame:
    work = df.copy()
    changed = True
    while changed and not work.empty:
        prev_len = len(work)

        user_counts = work.groupby('user_id').size()
        keep_users = set(user_counts[user_counts >= min_user_interactions].index)
        work = work[work['user_id'].isin(keep_users)].copy()

        item_counts = work.groupby('item_id').size()
        keep_items = set(item_counts[item_counts >= min_item_interactions].index)
        work = work[work['item_id'].isin(keep_items)].copy()

        changed = len(work) != prev_len
    return work


def summarize(df: pd.DataFrame) -> dict:
    user_counts = df.groupby('user_id').size()
    item_counts = df.groupby('item_id').size()
    return {
        'rows': int(len(df)),
        'users': int(user_counts.size),
        'items': int(item_counts.size),
        'avg_interactions_per_user': float(user_counts.mean()) if not user_counts.empty else 0.0,
        'median_interactions_per_user': float(user_counts.median()) if not user_counts.empty else 0.0,
        'avg_interactions_per_item': float(item_counts.mean()) if not item_counts.empty else 0.0,
        'median_interactions_per_item': float(item_counts.median()) if not item_counts.empty else 0.0,
        'users_lt5': int((user_counts < 5).sum()) if not user_counts.empty else 0,
        'items_lt3': int((item_counts < 3).sum()) if not item_counts.empty else 0,
        'positive_ratio_ge4': float((df['rating'] >= 4.0).mean()) if 'rating' in df.columns and len(df) else 0.0,
    }


def main() -> None:
    args = parse_args()
    if not args.source.exists():
        raise FileNotFoundError(f'missing source csv: {args.source}')

    df = pd.read_csv(args.source)
    df['user_id'] = df['user_id'].astype(str)
    df['item_id'] = df['item_id'].astype(str)
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df = df.dropna(subset=['user_id', 'item_id', 'rating', 'time']).copy()
    df['time'] = df['time'].astype('int64')
    df = df.sort_values(['user_id', 'time', 'item_id']).reset_index(drop=True)

    before = summarize(df)
    deduped = dedupe_dataframe(df, args.dedupe_key)
    after_dedup = summarize(deduped)
    cleaned = iterative_filter(deduped, args.min_user_interactions, args.min_item_interactions)
    cleaned = cleaned.sort_values(['user_id', 'time', 'item_id']).reset_index(drop=True)
    after_clean = summarize(cleaned)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(args.output, index=False, encoding='utf-8')

    print(f'source={args.source}')
    print(f'output={args.output}')
    print(f'dedupe_key={args.dedupe_key}')
    print(f'min_user_interactions={args.min_user_interactions}')
    print(f'min_item_interactions={args.min_item_interactions}')
    print(f'before={before}')
    print(f'after_dedup={after_dedup}')
    print(f'after_clean={after_clean}')


if __name__ == '__main__':
    main()
