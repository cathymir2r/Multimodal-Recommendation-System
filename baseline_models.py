import torch
import torch.nn as nn


class TextOnlyRecommender(nn.Module):
    def __init__(self, num_users: int, text_dim: int, user_emb_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, user_emb_dim)
        self.layers = nn.Sequential(
            nn.Linear(user_emb_dim + text_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, user_idx: torch.Tensor, text_vec: torch.Tensor, img_vec: torch.Tensor | None = None) -> torch.Tensor:
        features = torch.cat([self.user_emb(user_idx), text_vec], dim=-1)
        return self.layers(features).squeeze(-1)


class ImageOnlyRecommender(nn.Module):
    def __init__(self, num_users: int, img_dim: int, user_emb_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, user_emb_dim)
        self.layers = nn.Sequential(
            nn.Linear(user_emb_dim + img_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, user_idx: torch.Tensor, text_vec: torch.Tensor | None = None, img_vec: torch.Tensor | None = None) -> torch.Tensor:
        features = torch.cat([self.user_emb(user_idx), img_vec], dim=-1)
        return self.layers(features).squeeze(-1)


class EarlyFusionRecommender(nn.Module):
    def __init__(self, num_users: int, text_dim: int, img_dim: int, user_emb_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, user_emb_dim)
        self.layers = nn.Sequential(
            nn.Linear(user_emb_dim + text_dim + img_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, user_idx: torch.Tensor, text_vec: torch.Tensor, img_vec: torch.Tensor) -> torch.Tensor:
        features = torch.cat([self.user_emb(user_idx), text_vec, img_vec], dim=-1)
        return self.layers(features).squeeze(-1)


class GatedFusionRecommender(nn.Module):
    def __init__(self, num_users: int, text_dim: int, img_dim: int, user_emb_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, user_emb_dim)
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.img_proj = nn.Sequential(
            nn.Linear(img_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.layers = nn.Sequential(
            nn.Linear(user_emb_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, user_idx: torch.Tensor, text_vec: torch.Tensor, img_vec: torch.Tensor) -> torch.Tensor:
        text_hidden = self.text_proj(text_vec)
        img_hidden = self.img_proj(img_vec)
        gate = self.gate(torch.cat([text_hidden, img_hidden], dim=-1))
        fused = gate * text_hidden + (1.0 - gate) * img_hidden
        features = torch.cat([self.user_emb(user_idx), fused], dim=-1)
        return self.layers(features).squeeze(-1)
