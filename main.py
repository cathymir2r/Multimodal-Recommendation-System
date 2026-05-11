import json
import os
import re
import sys
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from preprocessing import build_item_corpus, clean_text
from experiment_config import ExperimentConfig


DEFAULT_DATA_CSV_PATH = 'amazon_clothing_with_images_balanced.csv' if os.path.exists('amazon_clothing_with_images_balanced.csv') else 'amazon_clothing.csv'
IMAGE_ROOT = '.'
OUTPUT_DIR = 'outputs'
IMAGE_CACHE_DIR = os.path.join(OUTPUT_DIR, 'feature_image_cache')

TEXT_MODEL_NAME = 'bert-base-uncased'
IMAGE_MODEL_NAME = 'resnet50'
MAX_TEXT_LEN = 96
TEXT_BATCH_SIZE = 16
IMAGE_BATCH_SIZE = 8
NUM_ROWS_FOR_ANALYSIS = 5000
NUM_ITEMS_FOR_FEATURES = None

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
IMAGE_OUTPUT_SIZE = 2048


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


class ItemCorpusDataset(Dataset):
    def __init__(self, item_df: pd.DataFrame, image_root: str):
        self.item_df = item_df.reset_index(drop=True)
        self.image_root = image_root

    def __len__(self) -> int:
        return len(self.item_df)

    def __getitem__(self, index: int) -> dict:
        row = self.item_df.iloc[index]
        image_path = str(row.get('image_path', '')).strip()
        if image_path and not os.path.isabs(image_path):
            image_path = os.path.join(self.image_root, image_path)
        image_url = str(row.get('image_url', '')).strip()
        return {
            'item_id': str(row['item_id']),
            'text': str(row['text']),
            'image_path': image_path,
            'image_url': image_url,
        }


def collate_fn(batch: List[dict]) -> List[dict]:
    return batch


def parse_image_candidates(raw_value: object) -> list[str]:
    if raw_value is None:
        return []
    if isinstance(raw_value, float) and pd.isna(raw_value):
        return []
    if isinstance(raw_value, list):
        return [str(item).strip() for item in raw_value if str(item).strip()]

    text_value = str(raw_value).strip()
    if not text_value or text_value.lower() == 'nan':
        return []
    if text_value.startswith('[') and text_value.endswith(']'):
        try:
            parsed = json.loads(text_value.replace("'", '"'))
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except json.JSONDecodeError:
            pass
    return [text_value]


def sanitize_item_id(item_id: str) -> str:
    return re.sub(r'[^a-zA-Z0-9_.-]+', '_', str(item_id))


def resolve_image_path(item_id: str, image_path: str, image_url: str) -> str:
    if image_path and os.path.exists(image_path):
        return image_path

    ensure_dir(IMAGE_CACHE_DIR)
    cache_path = os.path.join(IMAGE_CACHE_DIR, f'{sanitize_item_id(item_id)}.img')
    if os.path.exists(cache_path):
        return cache_path

    for candidate_url in parse_image_candidates(image_url):
        if not candidate_url:
            continue
        url = candidate_url
        if url.startswith('http://'):
            url = 'https://' + url[len('http://'):]
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=20)
            if response.status_code != 200 or not response.content:
                continue
            with open(cache_path, 'wb') as file:
                file.write(response.content)
            return cache_path
        except requests.RequestException:
            continue
    return ''


def basic_data_analysis(df: pd.DataFrame, output_dir: str) -> None:
    ensure_dir(output_dir)
    print('===== Data Info =====')
    print(df.info())
    print(df.head())

    missing = df.isnull().mean().sort_values(ascending=False)
    stats = {
        'num_records': int(len(df)),
        'num_users': int(df['user_id'].nunique()) if 'user_id' in df.columns else 0,
        'num_items': int(df['item_id'].nunique()) if 'item_id' in df.columns else 0,
    }
    with open(os.path.join(output_dir, 'basic_stats.json'), 'w', encoding='utf-8') as file:
        json.dump(stats, file, ensure_ascii=False, indent=2)

    plt.figure(figsize=(10, 5))
    missing.head(20).mul(100).plot(kind='bar')
    plt.ylabel('Missing Ratio (%)')
    plt.title('Top-20 Missing Value Ratios')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'missing_ratio_top20.png'))
    plt.close()

    if 'rating' in df.columns and not df['rating'].dropna().empty:
        plt.figure(figsize=(6, 4))
        sns.histplot(df['rating'].dropna(), bins=40, kde=True)
        plt.title('Distribution of rating')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'rating_dist.png'))
        plt.close()

    if 'category' in df.columns and not df['category'].dropna().empty:
        plt.figure(figsize=(10, 5))
        df['category'].dropna().value_counts().head(20).plot(kind='bar')
        plt.title('Top-20 category distribution')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'category_top20.png'))
        plt.close()


def load_text_model():
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
    model = AutoModel.from_pretrained(TEXT_MODEL_NAME)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model


def load_image_model():
    global IMAGE_OUTPUT_SIZE
    if IMAGE_MODEL_NAME == 'vit_b_16':
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads = torch.nn.Identity()
        IMAGE_OUTPUT_SIZE = 768
        weights = models.ViT_B_16_Weights.IMAGENET1K_V1
        transform = weights.transforms()
    else:
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        model.fc = torch.nn.Identity()
        IMAGE_OUTPUT_SIZE = 2048
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    model.to(DEVICE)
    model.eval()
    return model, transform


def get_text_embeddings(texts: List[str], tokenizer, model) -> np.ndarray:
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=MAX_TEXT_LEN, return_tensors='pt')
    encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
    return cls_embedding.cpu().numpy()


def get_image_embedding(image_path: str, model, transform) -> np.ndarray:
    if not image_path or not os.path.exists(image_path):
        return np.zeros((IMAGE_OUTPUT_SIZE,), dtype=np.float32)
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            embedding = model(image_tensor)
        return embedding.squeeze(0).cpu().numpy()
    except Exception as exc:
        print(f'failed to load image {image_path}: {exc}')
        return np.zeros((IMAGE_OUTPUT_SIZE,), dtype=np.float32)


def extract_text_features(item_df: pd.DataFrame, output_dir: str) -> None:
    ensure_dir(output_dir)
    tokenizer, model = load_text_model()
    dataset = ItemCorpusDataset(item_df, IMAGE_ROOT)
    loader = DataLoader(dataset, batch_size=TEXT_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    all_ids = []
    all_features = []
    for batch in tqdm(loader, desc='Extracting text features'):
        texts = [sample['text'] for sample in batch]
        item_ids = [sample['item_id'] for sample in batch]
        features = get_text_embeddings(texts, tokenizer, model)
        all_ids.extend(item_ids)
        all_features.append(features)

    np.save(os.path.join(output_dir, 'item_ids_text.npy'), np.array(all_ids))
    np.save(os.path.join(output_dir, 'text_features.npy'), np.concatenate(all_features, axis=0))
    with open(os.path.join(output_dir, 'metadata.json'), 'w', encoding='utf-8') as file:
        json.dump({'text_model_name': TEXT_MODEL_NAME, 'text_dim': int(np.concatenate(all_features, axis=0).shape[1])}, file, indent=2)


def extract_image_features(item_df: pd.DataFrame, output_dir: str) -> None:
    ensure_dir(output_dir)
    model, transform = load_image_model()
    dataset = ItemCorpusDataset(item_df, IMAGE_ROOT)
    loader = DataLoader(dataset, batch_size=IMAGE_BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

    all_ids = []
    all_features = []
    for batch in tqdm(loader, desc=f'Extracting image features ({IMAGE_MODEL_NAME})'):
        item_ids = [sample['item_id'] for sample in batch]
        paths = [
            resolve_image_path(sample['item_id'], sample['image_path'], sample.get('image_url', ''))
            for sample in batch
        ]
        batch_features = [get_image_embedding(path, model, transform) for path in paths]
        all_ids.extend(item_ids)
        all_features.append(np.stack(batch_features, axis=0))

    merged_features = np.concatenate(all_features, axis=0)
    np.save(os.path.join(output_dir, 'item_ids_image.npy'), np.array(all_ids))
    np.save(os.path.join(output_dir, 'image_features.npy'), merged_features)
    with open(os.path.join(output_dir, 'metadata.json'), 'w', encoding='utf-8') as file:
        json.dump({'image_model_name': IMAGE_MODEL_NAME, 'image_dim': int(merged_features.shape[1])}, file, indent=2)


def build_clean_item_corpus(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df.copy()
    for col in ['title', 'description', 'review_text']:
        if col in clean_df.columns:
            clean_df[col] = clean_df[col].apply(clean_text)
    item_df = build_item_corpus(clean_df)
    return item_df.drop_duplicates(subset=['item_id']).reset_index(drop=True)


def main() -> None:
    config = ExperimentConfig()
    if len(sys.argv) > 1:
        config.image_backbone = sys.argv[1].strip().lower()
    data_csv_path = config.data_csv_path
    if len(sys.argv) > 2:
        data_csv_path = sys.argv[2].strip()
    if not data_csv_path:
        data_csv_path = DEFAULT_DATA_CSV_PATH
    global IMAGE_MODEL_NAME
    IMAGE_MODEL_NAME = config.image_backbone

    if not os.path.exists(data_csv_path):
        raise FileNotFoundError(f'Missing data file: {data_csv_path}')

    ensure_dir(OUTPUT_DIR)
    config.ensure_dirs()
    df = pd.read_csv(data_csv_path)
    analysis_df = df.head(NUM_ROWS_FOR_ANALYSIS).copy() if NUM_ROWS_FOR_ANALYSIS else df.copy()
    basic_data_analysis(analysis_df, os.path.join(OUTPUT_DIR, 'analysis'))

    item_df = build_clean_item_corpus(df)
    if NUM_ITEMS_FOR_FEATURES:
        item_df = item_df.head(NUM_ITEMS_FOR_FEATURES).copy()

    item_df.to_csv(os.path.join(OUTPUT_DIR, 'item_corpus.csv'), index=False, encoding='utf-8')
    extract_text_features(item_df, config.text_feature_dir)
    extract_image_features(item_df, config.image_feature_dir)
    print(
        f'saved analysis and aligned features to {OUTPUT_DIR} '
        f'using data={data_csv_path} with image backbone={IMAGE_MODEL_NAME}'
    )


if __name__ == '__main__':
    main()
