import json
import os
from dataclasses import dataclass
from threading import Lock
from typing import Dict, List

import numpy as np
import pandas as pd
import requests
import torch
from django.conf import settings

from multimodal_models import SequenceAttentionRecommender
from preprocessing import prepare_interactions


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@dataclass
class RecommendationItem:
    item_id: str
    score: float
    title: str | None
    rating: float | None
    image_url: str | None
    has_image: bool


class RecommendationService:
    def __init__(self):
        self._lock = Lock()
        self._initialized = False
        self.model = None
        self.user2idx: Dict[str, int] = {}
        self.item_text_emb: Dict[str, np.ndarray] = {}
        self.item_img_emb: Dict[str, np.ndarray] = {}
        self.item_meta: pd.DataFrame | None = None
        self.user_histories: Dict[str, List[str]] = {}
        self.asin2imgurl: Dict[str, str] = {}
        self.text_dim = 0
        self.img_dim = 0
        self.max_seq_len = 20

    def _load_feature_dict(self, ids_path: str, feat_path: str) -> Dict[str, np.ndarray]:
        item_ids = [str(x) for x in np.load(ids_path, allow_pickle=True)]
        features = np.load(feat_path)
        return {item_id: feat.astype(np.float32) for item_id, feat in zip(item_ids, features)}

    def _build_user_histories(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        histories = {}
        for user_id, group in df.groupby('user_id', sort=False):
            ordered = group.sort_values('time')
            histories[str(user_id)] = ordered['item_id'].astype(str).tolist()[-self.max_seq_len:]
        return histories

    def initialize(self) -> None:
        if self._initialized:
            return
        with self._lock:
            if self._initialized:
                return

            config = settings.MULTIMODAL_CONFIG
            # PyTorch 2.6 defaults torch.load(weights_only=True), which breaks legacy checkpoints saved with metadata.
            checkpoint = torch.load(config['MODEL_SAVE_PATH'], map_location=DEVICE, weights_only=False)
            self.user2idx = checkpoint['user2idx']
            self.text_dim = checkpoint['text_dim']
            self.img_dim = checkpoint['img_dim']
            self.max_seq_len = checkpoint.get('max_seq_len', 20)

            self.item_text_emb = self._load_feature_dict(config['TEXT_FEAT_IDS_PATH'], config['TEXT_FEAT_PATH'])
            if os.path.exists(config['IMAGE_FEAT_IDS_PATH']) and os.path.exists(config['IMAGE_FEAT_PATH']):
                self.item_img_emb = self._load_feature_dict(config['IMAGE_FEAT_IDS_PATH'], config['IMAGE_FEAT_PATH'])
            else:
                self.item_img_emb = {}

            self.model = SequenceAttentionRecommender(
                num_users=len(self.user2idx),
                text_dim=self.text_dim,
                img_dim=self.img_dim,
                hidden_dim=checkpoint['hidden_dim'],
                user_emb_dim=checkpoint['user_emb_dim'],
                dropout=checkpoint.get('dropout', 0.1),
                sequence_encoder_type=checkpoint.get('sequence_encoder_type', 'transformer'),
                num_attention_heads=checkpoint.get('num_attention_heads', 4),
                num_transformer_layers=checkpoint.get('num_transformer_layers', 2),
                max_seq_len=self.max_seq_len,
            )
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(DEVICE)
            self.model.eval()

            df = pd.read_csv(config['DATA_CSV_PATH'])
            interactions = prepare_interactions(df)
            self.user_histories = self._build_user_histories(interactions)

            meta_cols = [col for col in ['item_id', 'title', 'rating', 'image_url', 'brand', 'category'] if col in df.columns]
            item_meta_df = df[meta_cols].drop_duplicates(subset=['item_id']).copy()
            item_meta_df['item_id'] = item_meta_df['item_id'].astype(str)
            self.item_meta = item_meta_df.set_index('item_id')

            os.makedirs(config['IMAGE_CACHE_DIR'], exist_ok=True)
            self._initialized = True


    @staticmethod
    def parse_image_candidates(raw_value) -> list[str]:
        if raw_value is None:
            return []
        if isinstance(raw_value, float) and np.isnan(raw_value):
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

    def get_preferred_image_url(self, item_id: str) -> str | None:
        self.initialize()
        meta_url = self.lookup_meta_image_url(item_id)
        if meta_url:
            return meta_url
        if self.item_meta is not None and item_id in self.item_meta.index:
            row = self.item_meta.loc[item_id]
            candidates = self.parse_image_candidates(row.get('image_url') if 'image_url' in row else None)
            if candidates:
                return candidates[0]
        return None

    def _build_history_tensors(self, history_ids: List[str]):
        history_text = np.zeros((1, self.max_seq_len, self.text_dim), dtype=np.float32)
        history_img = np.zeros((1, self.max_seq_len, self.img_dim), dtype=np.float32)
        history_mask = np.zeros((1, self.max_seq_len), dtype=np.float32)
        history_ids = history_ids[-self.max_seq_len:]
        start_idx = self.max_seq_len - len(history_ids)
        for offset, item_id in enumerate(history_ids):
            history_text[0, start_idx + offset] = self.item_text_emb.get(item_id, np.zeros(self.text_dim, dtype=np.float32))
            history_img[0, start_idx + offset] = self.item_img_emb.get(item_id, np.zeros(self.img_dim, dtype=np.float32))
            history_mask[0, start_idx + offset] = 1.0
        return (
            torch.tensor(history_text, dtype=torch.float32, device=DEVICE),
            torch.tensor(history_img, dtype=torch.float32, device=DEVICE),
            torch.tensor(history_mask, dtype=torch.float32, device=DEVICE),
        )

    def recommend(self, user_id: str, top_k: int = 10) -> List[RecommendationItem]:
        self.initialize()
        if self.model is None:
            raise RuntimeError('model is not initialized')
        if user_id not in self.user2idx:
            raise ValueError('user_id not found in training data')

        history_ids = self.user_histories.get(user_id, [])
        history_set = set(history_ids)
        history_text, history_img, history_mask = self._build_history_tensors(history_ids)
        item_ids = np.array(list(self.item_text_emb.keys()))
        target_text_np = np.stack([self.item_text_emb[item_id] for item_id in item_ids]).astype(np.float32)
        target_text = torch.tensor(target_text_np, dtype=torch.float32, device=DEVICE)
        if self.item_img_emb:
            target_img_np = np.stack([self.item_img_emb.get(item_id, np.zeros(self.img_dim, dtype=np.float32)) for item_id in item_ids]).astype(np.float32)
        else:
            target_img_np = np.zeros((len(item_ids), self.img_dim), dtype=np.float32)
        target_img = torch.tensor(target_img_np, dtype=torch.float32, device=DEVICE)

        history_text = history_text.repeat(len(item_ids), 1, 1)
        history_img = history_img.repeat(len(item_ids), 1, 1)
        history_mask = history_mask.repeat(len(item_ids), 1)
        user_idx = torch.tensor([self.user2idx[user_id]], dtype=torch.long, device=DEVICE).repeat(len(item_ids))

        with torch.no_grad():
            scores = self.model(user_idx, target_text, target_img, history_text, history_img, history_mask).cpu().numpy().astype(np.float32)

        if history_ids:
            hist_text_vectors = np.stack([
                self.item_text_emb.get(item_id, np.zeros(self.text_dim, dtype=np.float32))
                for item_id in history_ids
            ]).astype(np.float32)
            target_text_norm = np.linalg.norm(target_text_np, axis=1, keepdims=True)
            hist_text_norm = np.linalg.norm(hist_text_vectors, axis=1, keepdims=True)
            safe_target_text = target_text_np / np.clip(target_text_norm, 1e-6, None)
            safe_hist_text = hist_text_vectors / np.clip(hist_text_norm, 1e-6, None)
            text_similarity = safe_target_text @ safe_hist_text.T

            recency_weights = np.linspace(0.6, 1.0, num=len(history_ids), dtype=np.float32)
            recency_weights = recency_weights / np.clip(recency_weights.sum(), 1e-6, None)
            weighted_text_similarity = (text_similarity * recency_weights.reshape(1, -1)).sum(axis=1)
            neighbor_bonus = weighted_text_similarity.copy()

            if self.img_dim > 0 and self.item_img_emb:
                hist_img_vectors = np.stack([
                    self.item_img_emb.get(item_id, np.zeros(self.img_dim, dtype=np.float32))
                    for item_id in history_ids
                ]).astype(np.float32)
                target_img_norm = np.linalg.norm(target_img_np, axis=1, keepdims=True)
                hist_img_norm = np.linalg.norm(hist_img_vectors, axis=1, keepdims=True)
                safe_target_img = target_img_np / np.clip(target_img_norm, 1e-6, None)
                safe_hist_img = hist_img_vectors / np.clip(hist_img_norm, 1e-6, None)
                img_similarity = safe_target_img @ safe_hist_img.T
                weighted_img_similarity = (img_similarity * recency_weights.reshape(1, -1)).sum(axis=1)
                neighbor_bonus = 0.8 * weighted_text_similarity + 0.2 * weighted_img_similarity

            score_min = float(scores.min())
            score_max = float(scores.max())
            if score_max > score_min:
                normalized_scores = (scores - score_min) / (score_max - score_min)
            else:
                normalized_scores = np.zeros_like(scores)
            scores = 0.15 * normalized_scores + 0.85 * neighbor_bonus.astype(np.float32)

        ranked_indices = np.argsort(-scores)
        items = []
        fallback_items = []
        for idx in ranked_indices:
            item_id = str(item_ids[idx])
            if item_id in history_set:
                continue
            title = None
            rating = None
            if self.item_meta is not None and item_id in self.item_meta.index:
                row = self.item_meta.loc[item_id]
                title = str(row['title']) if 'title' in row and pd.notna(row['title']) else None
                rating = float(row['rating']) if 'rating' in row and pd.notna(row['rating']) else None
            preferred_image_url = self.get_preferred_image_url(item_id)
            item = RecommendationItem(
                item_id=item_id,
                score=float(scores[idx]),
                title=title,
                rating=rating,
                image_url=preferred_image_url or f'/api/image/{item_id}/',
                has_image=preferred_image_url is not None,
            )
            if preferred_image_url:
                items.append(item)
                if len(items) >= min(top_k, len(item_ids)):
                    break
            else:
                fallback_items.append(item)

        if not items:
            return fallback_items[: min(top_k, len(fallback_items))]
        return items

    def lookup_meta_image_url(self, item_id: str) -> str | None:
        self.initialize()
        if item_id in self.asin2imgurl:
            return self.asin2imgurl[item_id] or None
        meta_json_path = settings.MULTIMODAL_CONFIG['META_JSON_PATH']
        if not os.path.exists(meta_json_path):
            self.asin2imgurl[item_id] = ''
            return None
        with open(meta_json_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if str(obj.get('asin', '')) != item_id:
                    continue
                image_url = None
                high_res = obj.get('imageURLHighRes')
                if isinstance(high_res, list) and high_res:
                    image_url = high_res[0]
                if not image_url:
                    image_list = obj.get('imageURL')
                    if isinstance(image_list, list) and image_list:
                        image_url = image_list[0]
                if not image_url:
                    image_url = obj.get('imUrl') or obj.get('image_url')
                self.asin2imgurl[item_id] = str(image_url or '')
                return self.asin2imgurl[item_id] or None
        self.asin2imgurl[item_id] = ''
        return None

    def fetch_image(self, item_id: str):
        self.initialize()
        cache_dir = settings.MULTIMODAL_CONFIG['IMAGE_CACHE_DIR']
        media_map = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.webp': 'image/webp',
            '.gif': 'image/gif',
            '.svg': 'image/svg+xml',
        }
        for ext, media_type in media_map.items():
            cached_path = os.path.join(cache_dir, f'{item_id}{ext}')
            if os.path.exists(cached_path):
                return cached_path, media_type

        url = self.get_preferred_image_url(item_id)
        if not url:
            return None, 'image/svg+xml'
        if url.startswith('http://'):
            url = 'https://' + url[len('http://'):]
        try:
            response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        except requests.RequestException:
            return None, 'image/svg+xml'
        if response.status_code != 200 or not response.content:
            return None, 'image/svg+xml'

        media_type = response.headers.get('Content-Type', '').split(';')[0].strip().lower()
        if media_type not in media_map.values():
            content = response.content
            if content.startswith(bytes.fromhex('89504E47')):
                media_type = 'image/png'
            elif content[:3] == b'GIF':
                media_type = 'image/gif'
            elif content[:4] == b'RIFF' and b'WEBP' in content[:16]:
                media_type = 'image/webp'
            elif content.startswith(bytes.fromhex('FFD8')):
                media_type = 'image/jpeg'
            else:
                media_type = 'image/jpeg'

        ext = next((suffix for suffix, mt in media_map.items() if mt == media_type), '.jpg')
        cache_path = os.path.join(cache_dir, f'{item_id}{ext}')
        for old_ext in media_map:
            old_path = os.path.join(cache_dir, f'{item_id}{old_ext}')
            if old_path != cache_path and os.path.exists(old_path):
                try:
                    os.remove(old_path)
                except OSError:
                    pass
        with open(cache_path, 'wb') as file:
            file.write(response.content)
        return cache_path, media_type


recommendation_service = RecommendationService()
