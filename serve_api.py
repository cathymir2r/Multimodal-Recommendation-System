import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
import requests
import torch

from multimodal_models import SequenceAttentionRecommender
from preprocessing import prepare_interactions


DATA_CSV_PATH = 'amazon_clothing.csv'
META_JSON_PATH = 'meta_Clothing_Shoes_and_Jewelry.json'
TEXT_FEAT_IDS_PATH = os.path.join('outputs', 'text_features', 'item_ids_text.npy')
TEXT_FEAT_PATH = os.path.join('outputs', 'text_features', 'text_features.npy')
IMAGE_FEAT_IDS_PATH = os.path.join('outputs', 'image_features', 'item_ids_image.npy')
IMAGE_FEAT_PATH = os.path.join('outputs', 'image_features', 'image_features.npy')
MODEL_SAVE_PATH = os.path.join('outputs', 'models', 'sequence_attention_recommender.pt')
IMAGE_CACHE_DIR = os.path.join('outputs', 'image_cache')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class RecommendRequest(BaseModel):
    user_id: str
    top_k: int = 10


class ItemInfo(BaseModel):
    item_id: str
    score: float
    title: str | None = None
    rating: float | None = None
    image_url: str | None = None


class RecommendResponse(BaseModel):
    user_id: str
    top_k: int
    items: List[ItemInfo]


app = FastAPI(title='Multi-Modal Clothing Recommender API')

model: SequenceAttentionRecommender | None = None
user2idx: Dict[str, int] = {}
item_text_emb: Dict[str, np.ndarray] = {}
item_img_emb: Dict[str, np.ndarray] = {}
item_meta: pd.DataFrame | None = None
user_histories: Dict[str, List[str]] = {}
img_dim = 0
text_dim = 0
max_seq_len = 20
asin2imgurl: Dict[str, str] = {}


def load_feature_dict(ids_path: str, feat_path: str) -> Dict[str, np.ndarray]:
    item_ids = [str(x) for x in np.load(ids_path, allow_pickle=True)]
    features = np.load(feat_path)
    return {item_id: feat.astype(np.float32) for item_id, feat in zip(item_ids, features)}


def lookup_meta_image_url(item_id: str) -> str | None:
    if item_id in asin2imgurl:
        return asin2imgurl[item_id] or None
    if not os.path.exists(META_JSON_PATH):
        asin2imgurl[item_id] = ''
        return None
    with open(META_JSON_PATH, 'r', encoding='utf-8') as file:
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
            asin2imgurl[item_id] = str(image_url or '')
            return asin2imgurl[item_id] or None
    asin2imgurl[item_id] = ''
    return None


def build_user_histories(df: pd.DataFrame, seq_len: int) -> Dict[str, List[str]]:
    histories = {}
    for user_id, group in df.groupby('user_id', sort=False):
        ordered = group.sort_values('time')
        histories[str(user_id)] = ordered['item_id'].astype(str).tolist()[-seq_len:]
    return histories


@app.on_event('startup')
def load_all():
    global model, user2idx, item_text_emb, item_img_emb, item_meta, user_histories, img_dim, text_dim, max_seq_len
    if not os.path.exists(MODEL_SAVE_PATH):
        raise RuntimeError(f'Missing model checkpoint: {MODEL_SAVE_PATH}')
    checkpoint = torch.load(MODEL_SAVE_PATH, map_location=DEVICE)
    user2idx = checkpoint['user2idx']
    text_dim = checkpoint['text_dim']
    img_dim = checkpoint['img_dim']
    max_seq_len = checkpoint.get('max_seq_len', 20)

    item_text_emb = load_feature_dict(TEXT_FEAT_IDS_PATH, TEXT_FEAT_PATH)
    item_img_emb = load_feature_dict(IMAGE_FEAT_IDS_PATH, IMAGE_FEAT_PATH) if os.path.exists(IMAGE_FEAT_IDS_PATH) else {}

    model = SequenceAttentionRecommender(
        num_users=len(user2idx),
        text_dim=text_dim,
        img_dim=img_dim,
        hidden_dim=checkpoint['hidden_dim'],
        user_emb_dim=checkpoint['user_emb_dim'],
        dropout=checkpoint.get('dropout', 0.1),
        sequence_encoder_type=checkpoint.get('sequence_encoder_type', 'transformer'),
        num_attention_heads=checkpoint.get('num_attention_heads', 4),
        num_transformer_layers=checkpoint.get('num_transformer_layers', 2),
        max_seq_len=max_seq_len,
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval()

    df = pd.read_csv(DATA_CSV_PATH)
    interactions = prepare_interactions(df)
    user_histories = build_user_histories(interactions, max_seq_len)

    meta_cols = [col for col in ['item_id', 'title', 'rating', 'image_url'] if col in df.columns]
    item_meta_df = df[meta_cols].drop_duplicates(subset=['item_id']).copy()
    item_meta_df['item_id'] = item_meta_df['item_id'].astype(str)
    item_meta = item_meta_df.set_index('item_id')
    os.makedirs(IMAGE_CACHE_DIR, exist_ok=True)


def build_history_tensors(history_ids: List[str]):
    history_text = np.zeros((1, max_seq_len, text_dim), dtype=np.float32)
    history_img = np.zeros((1, max_seq_len, img_dim), dtype=np.float32)
    history_mask = np.zeros((1, max_seq_len), dtype=np.float32)
    history_ids = history_ids[-max_seq_len:]
    start_idx = max_seq_len - len(history_ids)
    for offset, item_id in enumerate(history_ids):
        history_text[0, start_idx + offset] = item_text_emb.get(item_id, np.zeros(text_dim, dtype=np.float32))
        history_img[0, start_idx + offset] = item_img_emb.get(item_id, np.zeros(img_dim, dtype=np.float32))
        history_mask[0, start_idx + offset] = 1.0
    return (
        torch.tensor(history_text, dtype=torch.float32, device=DEVICE),
        torch.tensor(history_img, dtype=torch.float32, device=DEVICE),
        torch.tensor(history_mask, dtype=torch.float32, device=DEVICE),
    )


def recommend_for_user(user_id: str, top_k: int = 10) -> List[ItemInfo]:
    if model is None:
        raise RuntimeError('Model not loaded')
    if user_id not in user2idx:
        raise HTTPException(status_code=400, detail='user_id not found in training data')

    history_ids = user_histories.get(user_id, [])
    history_text, history_img, history_mask = build_history_tensors(history_ids)
    item_ids = np.array(list(item_text_emb.keys()))
    target_text = torch.tensor(np.stack([item_text_emb[item_id] for item_id in item_ids]), dtype=torch.float32, device=DEVICE)
    if item_img_emb:
        target_img_np = np.stack([item_img_emb.get(item_id, np.zeros(img_dim, dtype=np.float32)) for item_id in item_ids])
    else:
        target_img_np = np.zeros((len(item_ids), img_dim), dtype=np.float32)
    target_img = torch.tensor(target_img_np, dtype=torch.float32, device=DEVICE)

    history_text = history_text.repeat(len(item_ids), 1, 1)
    history_img = history_img.repeat(len(item_ids), 1, 1)
    history_mask = history_mask.repeat(len(item_ids), 1)
    user_idx = torch.tensor([user2idx[user_id]], dtype=torch.long, device=DEVICE).repeat(len(item_ids))

    with torch.no_grad():
        scores = model(user_idx, target_text, target_img, history_text, history_img, history_mask).cpu().numpy()

    ranked_indices = np.argsort(-scores)[: min(top_k, len(item_ids))]
    items = []
    for idx in ranked_indices:
        item_id = str(item_ids[idx])
        title = None
        rating = None
        if item_meta is not None and item_id in item_meta.index:
            row = item_meta.loc[item_id]
            title = str(row['title']) if 'title' in row and pd.notna(row['title']) else None
            rating = float(row['rating']) if 'rating' in row and pd.notna(row['rating']) else None
        items.append(ItemInfo(item_id=item_id, score=float(scores[idx]), title=title, rating=rating, image_url=f'/image/{item_id}'))
    return items


@app.get('/', response_class=HTMLResponse)
def index():
    return '<html><body><h1>FastAPI demo is still available.</h1><p>Use Django + Vue for the full system.</p></body></html>'


@app.get('/image/{item_id}')
def image(item_id: str):
    cache_path = os.path.join(IMAGE_CACHE_DIR, f'{item_id}.jpg')
    if os.path.exists(cache_path):
        return FileResponse(cache_path, media_type='image/jpeg')
    url = lookup_meta_image_url(str(item_id))
    if not url:
        svg = "<svg xmlns='http://www.w3.org/2000/svg' width='300' height='300'><rect width='100%25' height='100%25' fill='%23ddd'/><text x='50%25' y='50%25' dominant-baseline='middle' text-anchor='middle' fill='%23666' font-size='18'>No Image</text></svg>"
        return HTMLResponse(content=svg, media_type='image/svg+xml')
    if url.startswith('http://'):
        url = 'https://' + url[len('http://'):]
    try:
        resp = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        if resp.status_code != 200 or not resp.content:
            raise RuntimeError('image download failed')
        with open(cache_path, 'wb') as file:
            file.write(resp.content)
        return FileResponse(cache_path, media_type='image/jpeg')
    except Exception:
        svg = "<svg xmlns='http://www.w3.org/2000/svg' width='300' height='300'><rect width='100%25' height='100%25' fill='%23ddd'/><text x='50%25' y='50%25' dominant-baseline='middle' text-anchor='middle' fill='%23666' font-size='18'>Image Error</text></svg>"
        return HTMLResponse(content=svg, media_type='image/svg+xml')


@app.post('/recommend', response_model=RecommendResponse)
def recommend(req: RecommendRequest):
    items = recommend_for_user(req.user_id, req.top_k)
    return RecommendResponse(user_id=req.user_id, top_k=req.top_k, items=items)
