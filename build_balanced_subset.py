import csv
import sys
from collections import defaultdict
from pathlib import Path

csv.field_size_limit(10 ** 7)

SOURCE_CSV = Path('amazon_clothing_with_images.csv')
OUTPUT_CSV = Path('amazon_clothing_with_images_balanced.csv')
MAX_ITEMS = 5000
MAX_ROWS = 200000
MAX_BEHAVIORS_PER_ITEM = 40


def has_image(row: dict) -> bool:
    value = str(row.get('image_url', '') or '').strip()
    return bool(value and value.lower() != 'nan')


def main() -> None:
    if not SOURCE_CSV.exists():
        raise FileNotFoundError(f'missing source csv: {SOURCE_CSV}')

    selected_items: set[str] = set()
    per_item_count: dict[str, int] = defaultdict(int)
    written_rows = 0

    with SOURCE_CSV.open('r', encoding='utf-8', newline='') as src, OUTPUT_CSV.open('w', encoding='utf-8', newline='') as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()

        for row in reader:
            if not has_image(row):
                continue
            item_id = str(row.get('item_id', '')).strip()
            if not item_id:
                continue
            if item_id not in selected_items:
                if len(selected_items) >= MAX_ITEMS:
                    if written_rows >= MAX_ROWS:
                        break
                    continue
                selected_items.add(item_id)
            if per_item_count[item_id] >= MAX_BEHAVIORS_PER_ITEM:
                if len(selected_items) >= MAX_ITEMS and written_rows >= MAX_ROWS:
                    break
                continue
            writer.writerow(row)
            per_item_count[item_id] += 1
            written_rows += 1
            if len(selected_items) >= MAX_ITEMS and written_rows >= MAX_ROWS:
                break

    print(f'generated balanced csv: {OUTPUT_CSV}')
    print(f'selected_items={len(selected_items)}')
    print(f'written_rows={written_rows}')


if __name__ == '__main__':
    main()
