from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalFusionEncoder(nn.Module):
    def __init__(self, text_dim: int, img_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.cross_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_vec: torch.Tensor, img_vec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_hidden = self.text_proj(text_vec)
        img_hidden = self.img_proj(img_vec)
        gate = self.cross_gate(torch.cat([text_hidden, img_hidden], dim=-1))
        fused = gate * text_hidden + (1.0 - gate) * img_hidden
        return self.dropout(fused), text_hidden, img_hidden


class SequenceAttentionRecommender(nn.Module):
    def __init__(
        self,
        num_users: int,
        text_dim: int,
        img_dim: int,
        hidden_dim: int = 256,
        user_emb_dim: int = 64,
        num_gru_layers: int = 1,
        dropout: float = 0.1,
        sequence_encoder_type: str = 'transformer',
        num_attention_heads: int = 4,
        num_transformer_layers: int = 2,
        max_seq_len: int = 20,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        self.sequence_encoder_type = sequence_encoder_type.lower()
        self.user_emb = nn.Embedding(num_users, user_emb_dim)
        self.item_encoder = MultiModalFusionEncoder(text_dim=text_dim, img_dim=img_dim, hidden_dim=hidden_dim, dropout=dropout)

        if self.sequence_encoder_type == 'gru':
            self.history_encoder = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=num_gru_layers,
                batch_first=True,
            )
        else:
            self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_attention_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                batch_first=True,
                activation='gelu',
            )
            self.history_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Sequential(
            nn.Linear(user_emb_dim + hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def _encode_history(self, history_text: torch.Tensor, history_img: torch.Tensor, history_mask: torch.Tensor):
        batch_size, seq_len, _ = history_text.shape
        flat_text = history_text.reshape(batch_size * seq_len, -1)
        flat_img = history_img.reshape(batch_size * seq_len, -1)
        fused_history, text_hidden, img_hidden = self.item_encoder(flat_text, flat_img)
        fused_history = fused_history.reshape(batch_size, seq_len, -1)
        text_hidden = text_hidden.reshape(batch_size, seq_len, -1)
        img_hidden = img_hidden.reshape(batch_size, seq_len, -1)

        if self.sequence_encoder_type == 'gru':
            encoded_history, _ = self.history_encoder(fused_history)
            encoded_history = encoded_history * history_mask.unsqueeze(-1)
        else:
            positions = torch.arange(seq_len, device=history_text.device).unsqueeze(0).expand(batch_size, seq_len)
            encoded_input = fused_history + self.position_embedding(positions)
            padding_mask = history_mask == 0
            encoded_history = self.history_encoder(encoded_input, src_key_padding_mask=padding_mask)
            encoded_history = encoded_history * history_mask.unsqueeze(-1)

        return encoded_history, text_hidden, img_hidden

    def _alignment_loss(self, text_hidden: torch.Tensor, img_hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normalized_text = F.normalize(text_hidden, dim=-1)
        normalized_img = F.normalize(img_hidden, dim=-1)
        cosine = (normalized_text * normalized_img).sum(dim=-1)
        loss = 1.0 - cosine
        return (loss * mask).sum() / mask.sum().clamp_min(1.0)

    def encode_history_state(
        self,
        history_text: torch.Tensor,
        history_img: torch.Tensor,
        history_mask: torch.Tensor,
    ):
        history_hidden, _, _ = self._encode_history(history_text, history_img, history_mask)
        keys = self.key_proj(history_hidden)
        values = self.value_proj(history_hidden)
        return keys, values, history_mask

    def score_with_history_state(
        self,
        user_idx: torch.Tensor,
        target_text: torch.Tensor,
        target_img: torch.Tensor,
        history_state,
    ) -> torch.Tensor:
        keys, values, history_mask = history_state
        batch_size = target_text.size(0)

        if keys.size(0) == 1 and batch_size > 1:
            keys = keys.expand(batch_size, -1, -1)
            values = values.expand(batch_size, -1, -1)
            history_mask = history_mask.expand(batch_size, -1)

        target_fused, _, _ = self.item_encoder(target_text, target_img)
        query = self.query_proj(target_fused).unsqueeze(1)
        attn_logits = torch.matmul(query, keys.transpose(1, 2)).squeeze(1)
        attn_logits = attn_logits.masked_fill(history_mask == 0, -1e9)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_context = torch.bmm(attn_weights.unsqueeze(1), values).squeeze(1)

        user_emb = self.user_emb(user_idx)
        features = torch.cat([user_emb, target_fused, attn_context], dim=-1)
        return self.output(features).squeeze(-1)

    def forward(
        self,
        user_idx: torch.Tensor,
        target_text: torch.Tensor,
        target_img: torch.Tensor,
        history_text: torch.Tensor,
        history_img: torch.Tensor,
        history_mask: torch.Tensor,
        return_aux: bool = False,
    ):
        target_fused, target_text_hidden, target_img_hidden = self.item_encoder(target_text, target_img)
        history_hidden, history_text_hidden, history_img_hidden = self._encode_history(history_text, history_img, history_mask)
        keys = self.key_proj(history_hidden)
        values = self.value_proj(history_hidden)
        query = self.query_proj(target_fused).unsqueeze(1)
        attn_logits = torch.matmul(query, keys.transpose(1, 2)).squeeze(1)
        attn_logits = attn_logits.masked_fill(history_mask == 0, -1e9)
        attn_weights = torch.softmax(attn_logits, dim=-1)
        attn_context = torch.bmm(attn_weights.unsqueeze(1), values).squeeze(1)

        user_emb = self.user_emb(user_idx)
        features = torch.cat([user_emb, target_fused, attn_context], dim=-1)
        pred = self.output(features).squeeze(-1)

        if not return_aux:
            return pred

        target_alignment = 1.0 - F.cosine_similarity(
            F.normalize(target_text_hidden, dim=-1),
            F.normalize(target_img_hidden, dim=-1),
            dim=-1,
        )
        target_alignment = target_alignment.mean()
        history_alignment = self._alignment_loss(history_text_hidden, history_img_hidden, history_mask)
        aux = {
            'alignment_loss': 0.5 * (target_alignment + history_alignment),
            'attention_entropy': (-(attn_weights * torch.log(attn_weights.clamp_min(1e-9))).sum(dim=-1)).mean(),
        }
        return pred, aux
