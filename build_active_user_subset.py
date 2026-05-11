import argparse
import csv
from collections import Counter
from pathlib import Path


csv.field_size_limit(10 ** 7)

SOURCE_CSV = Path('amazon_clothing_with_images.csv')
DEFAULT_OUTPUT = Path('outputs/subsets/amazon_clothing_top500_users_ge3.csv')


def has_image(row: dict) -> bool:
    value = str(row.get('image_url', '') or '').strip()
    return bool(value and value.lower() != 'nan')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build a user-centric subset for sequential recommendation.')
    parser.add_argument('--source', type=Path, default=SOURCE_CSV, help='Source CSV with image_url column.')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='Output subset CSV path.')
    parser.add_argument('--min-user-interactions', type=int, default=3, help='Keep users with at least this many interactions.')
    parser.add_argument('--max-users', type=int, default=500, help='Keep at most this many most-active users. Use 0 for all.')
    parser.add_argument('--min-item-interactions', type=int, default=1, help='Keep items with at least this many interactions from selected users.')
    parser.add_argument('--max-items', type=int, default=5000, help='Keep at most this many most-frequent items. Use 0 for all.')
    parser.add_argument('--max-rows-per-user', type=int, default=0, help='Optional cap per user, keeping the latest rows only. Use 0 for all.')
    return parser.parse_args()


def count_user_interactions(source_path: Path) -> Counter:
    user_counts: Counter = Counter()
    with source_path.open('r', encoding='utf-8', newline='') as src:
        reader = csv.DictReader(src)
        for row in reader:
            if not has_image(row):
                continue
            user_id = str(row.get('user_id', '')).strip()
            if user_id:
                user_counts[user_id] += 1
    return user_counts


def select_users(user_counts: Counter, min_user_interactions: int, max_users: int) -> set[str]:
    filtered = [
        (user_id, count)
        for user_id, count in user_counts.items()
        if count >= min_user_interactions
    ]
    filtered.sort(key=lambda pair: (-pair[1], pair[0]))
    if max_users > 0:
        filtered = filtered[:max_users]
    return {user_id for user_id, _ in filtered}


def count_item_interactions(source_path: Path, selected_users: set[str]) -> Counter:
    item_counts: Counter = Counter()
    with source_path.open('r', encoding='utf-8', newline='') as src:
        reader = csv.DictReader(src)
        for row in reader:
            if not has_image(row):
                continue
            user_id = str(row.get('user_id', '')).strip()
            if user_id not in selected_users:
                continue
            item_id = str(row.get('item_id', '')).strip()
            if item_id:
                item_counts[item_id] += 1
    return item_counts


def select_items(item_counts: Counter, min_item_interactions: int, max_items: int) -> set[str]:
    filtered = [
        (item_id, count)
        for item_id, count in item_counts.items()
        if count >= min_item_interactions
    ]
    filtered.sort(key=lambda pair: (-pair[1], pair[0]))
    if max_items > 0:
        filtered = filtered[:max_items]
    return {item_id for item_id, _ in filtered}


def write_subset(
    source_path: Path,
    output_path: Path,
    selected_users: set[str],
    selected_items: set[str],
    max_rows_per_user: int,
) -> tuple[int, int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    written_rows = 0
    per_user_rows: dict[str, list[dict]] = {}
    kept_items: set[str] = set()

    with source_path.open('r', encoding='utf-8', newline='') as src:
        reader = csv.DictReader(src)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError('source csv has no header')

        for row in reader:
            if not has_image(row):
                continue
            user_id = str(row.get('user_id', '')).strip()
            if user_id not in selected_users:
                continue
            item_id = str(row.get('item_id', '')).strip()
            if item_id not in selected_items:
                continue
            per_user_rows.setdefault(user_id, []).append(row)
            kept_items.add(item_id)

    with output_path.open('w', encoding='utf-8', newline='') as dst:
        writer = csv.DictWriter(dst, fieldnames=fieldnames)
        writer.writeheader()
        for user_id in sorted(per_user_rows.keys()):
            rows = per_user_rows[user_id]
            if max_rows_per_user > 0 and len(rows) > max_rows_per_user:
                rows = rows[-max_rows_per_user:]
            for row in rows:
                writer.writerow(row)
                written_rows += 1

    return len(per_user_rows), written_rows, len(kept_items)


def main() -> None:
    args = parse_args()
    if not args.source.exists():
        raise FileNotFoundError(f'missing source csv: {args.source}')

    user_counts = count_user_interactions(args.source)
    selected_users = select_users(user_counts, args.min_user_interactions, args.max_users)
    item_counts = count_item_interactions(args.source, selected_users)
    selected_items = select_items(item_counts, args.min_item_interactions, args.max_items)
    kept_users, written_rows, kept_items = write_subset(
        args.source,
        args.output,
        selected_users,
        selected_items,
        args.max_rows_per_user,
    )

    print(f'generated user-centric subset: {args.output}')
    print(f'min_user_interactions={args.min_user_interactions}')
    print(f'max_users={args.max_users}')
    print(f'min_item_interactions={args.min_item_interactions}')
    print(f'max_items={args.max_items}')
    print(f'max_rows_per_user={args.max_rows_per_user}')
    print(f'selected_users={kept_users}')
    print(f'selected_items={kept_items}')
    print(f'written_rows={written_rows}')


if __name__ == '__main__':
    main()
