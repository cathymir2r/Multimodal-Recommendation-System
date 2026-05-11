import csv
import json
from pathlib import Path
from typing import Any

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent
REVIEWS_JSON = PROJECT_ROOT / 'Clothing_Shoes_and_Jewelry.json'
META_JSON = PROJECT_ROOT / 'meta_Clothing_Shoes_and_Jewelry.json'
OUT_CSV = PROJECT_ROOT / 'amazon_clothing_with_images.csv'

FIELDNAMES = [
    'user_id',
    'item_id',
    'title',
    'description',
    'review_text',
    'image_path',
    'brand',
    'price',
    'category',
    'rating',
    'time',
    'image_url',
]


def normalize_text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, list):
        return ' '.join(normalize_text(item) for item in value if normalize_text(item))
    text = str(value).strip()
    if text.lower() == 'nan':
        return ''
    return text


def pick_image_url(meta: dict) -> str:
    for key in ['imageURLHighRes', 'imageURL', 'images']:
        value = meta.get(key)
        if isinstance(value, list) and value:
            first = value[0]
            if isinstance(first, dict):
                for nested_key in ['large', 'hi_res', 'url']:
                    nested = normalize_text(first.get(nested_key))
                    if nested:
                        return nested
            text = normalize_text(first)
            if text:
                return text
    for key in ['imUrl', 'image_url']:
        text = normalize_text(meta.get(key))
        if text:
            return text
    return ''


def build_meta_lookup(meta_path: Path) -> dict[str, dict[str, str]]:
    asin_to_meta: dict[str, dict[str, str]] = {}
    with meta_path.open('r', encoding='utf-8') as handle:
        for line in tqdm(handle, desc='loading meta'):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = normalize_text(obj.get('asin'))
            if not asin:
                continue
            categories = obj.get('categories')
            category_text = ''
            if isinstance(categories, list) and categories:
                first = categories[0]
                category_text = normalize_text(first if not isinstance(first, list) else ' > '.join(normalize_text(part) for part in first))
            asin_to_meta[asin] = {
                'title': normalize_text(obj.get('title')),
                'description': normalize_text(obj.get('description')),
                'brand': normalize_text(obj.get('brand')),
                'price': normalize_text(obj.get('price')),
                'category': category_text,
                'image_url': pick_image_url(obj),
            }
    return asin_to_meta


def build_csv() -> None:
    if not REVIEWS_JSON.exists():
        raise FileNotFoundError(f'missing reviews json: {REVIEWS_JSON}')
    if not META_JSON.exists():
        raise FileNotFoundError(f'missing meta json: {META_JSON}')

    asin_to_meta = build_meta_lookup(META_JSON)
    with REVIEWS_JSON.open('r', encoding='utf-8') as review_handle, OUT_CSV.open('w', encoding='utf-8', newline='') as out_handle:
        writer = csv.DictWriter(out_handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for line in tqdm(review_handle, desc='building csv'):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            item_id = normalize_text(obj.get('asin'))
            meta = asin_to_meta.get(item_id, {})
            title = normalize_text(obj.get('summary')) or meta.get('title', '')
            writer.writerow({
                'user_id': normalize_text(obj.get('reviewerID')),
                'item_id': item_id,
                'title': title,
                'description': meta.get('description', ''),
                'review_text': normalize_text(obj.get('reviewText')),
                'image_path': '',
                'brand': meta.get('brand', ''),
                'price': meta.get('price', ''),
                'category': meta.get('category', '') or 'Clothing_Shoes_and_Jewelry',
                'rating': obj.get('overall', ''),
                'time': obj.get('unixReviewTime', ''),
                'image_url': meta.get('image_url', ''),
            })

    print(f'generated csv: {OUT_CSV}')


if __name__ == '__main__':
    build_csv()
