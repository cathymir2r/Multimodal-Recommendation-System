import os
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ExperimentConfig:
    data_csv_path: str = 'amazon_clothing_with_images_balanced.csv' if Path('amazon_clothing_with_images_balanced.csv').exists() else 'amazon_clothing.csv'
    outputs_dir: str = 'outputs'
    text_feat_ids_path: str = 'outputs/text_features/item_ids_text.npy'
    text_feat_path: str = 'outputs/text_features/text_features.npy'
    image_feat_ids_path: str = 'outputs/image_features/item_ids_image.npy'
    image_feat_path: str = 'outputs/image_features/image_features.npy'
    model_dir: str = 'outputs/models'
    reports_dir: str = 'outputs/reports'
    batch_size: int = 64
    num_epochs: int = 5
    learning_rate: float = 1e-3
    max_seq_len: int = 20
    min_history_len: int = 0
    hidden_dim: int = 256
    user_emb_dim: int = 64
    test_ratio: float = 0.2
    positive_threshold: float = 4.0
    dropout: float = 0.1
    image_backbone: str = 'resnet50'
    sequence_encoder_type: str = 'transformer'
    num_attention_heads: int = 4
    num_transformer_layers: int = 2
    alignment_weight: float = 0.05
    random_seed: int = 42
    loss_type: str = "mse"
    selection_metric: str = "mse"
    selection_min_epoch: int = 1
    training_objective: str = "pointwise"
    ranking_eval_mode: str = "full"
    ranking_num_negatives: int = 99
    ranking_item_batch_size: int = 256

    def __post_init__(self) -> None:
        env_map = {
            "EXP_DATA_CSV_PATH": ("data_csv_path", str),
            "EXP_OUTPUTS_DIR": ("outputs_dir", str),
            "EXP_MODEL_DIR": ("model_dir", str),
            "EXP_REPORTS_DIR": ("reports_dir", str),
            "EXP_BATCH_SIZE": ("batch_size", int),
            "EXP_NUM_EPOCHS": ("num_epochs", int),
            "EXP_LEARNING_RATE": ("learning_rate", float),
            "EXP_MAX_SEQ_LEN": ("max_seq_len", int),
            "EXP_MIN_HISTORY_LEN": ("min_history_len", int),
            "EXP_HIDDEN_DIM": ("hidden_dim", int),
            "EXP_USER_EMB_DIM": ("user_emb_dim", int),
            "EXP_POSITIVE_THRESHOLD": ("positive_threshold", float),
            "EXP_DROPOUT": ("dropout", float),
            "EXP_IMAGE_BACKBONE": ("image_backbone", str),
            "EXP_SEQUENCE_ENCODER_TYPE": ("sequence_encoder_type", str),
            "EXP_ALIGNMENT_WEIGHT": ("alignment_weight", float),
            "EXP_RANDOM_SEED": ("random_seed", int),
            "EXP_LOSS_TYPE": ("loss_type", str),
            "EXP_SELECTION_METRIC": ("selection_metric", str),
            "EXP_SELECTION_MIN_EPOCH": ("selection_min_epoch", int),
            "EXP_TRAINING_OBJECTIVE": ("training_objective", str),
            "EXP_RANKING_EVAL_MODE": ("ranking_eval_mode", str),
            "EXP_RANKING_NUM_NEGATIVES": ("ranking_num_negatives", int),
            "EXP_RANKING_ITEM_BATCH_SIZE": ("ranking_item_batch_size", int),
        }

        for env_key, (field_name, caster) in env_map.items():
            raw_value = os.getenv(env_key)
            if raw_value in (None, ""):
                continue
            setattr(self, field_name, caster(raw_value))

    @staticmethod
    def _norm_path(path: str) -> str:
        return str(Path(path)) if path else ''

    def ensure_dirs(self) -> None:
        Path(self.outputs_dir).mkdir(parents=True, exist_ok=True)
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        Path(self.reports_dir).mkdir(parents=True, exist_ok=True)
        Path(self.text_feature_dir).mkdir(parents=True, exist_ok=True)
        Path(self.image_feature_dir).mkdir(parents=True, exist_ok=True)

    @property
    def text_feature_dir(self) -> str:
        return str(Path(self.outputs_dir) / 'text_features')

    @property
    def image_feature_dir(self) -> str:
        return str(Path(self.outputs_dir) / f'image_features_{self.image_backbone}')

    @property
    def text_feat_ids_path_resolved(self) -> str:
        default_path = str(Path(self.text_feature_dir) / 'item_ids_text.npy')
        legacy_path = str(Path(self.outputs_dir) / 'text_features' / 'item_ids_text.npy')
        if not self.text_feat_ids_path or self._norm_path(self.text_feat_ids_path) == self._norm_path(legacy_path):
            return default_path
        return self.text_feat_ids_path

    @property
    def text_feat_path_resolved(self) -> str:
        default_path = str(Path(self.text_feature_dir) / 'text_features.npy')
        legacy_path = str(Path(self.outputs_dir) / 'text_features' / 'text_features.npy')
        if not self.text_feat_path or self._norm_path(self.text_feat_path) == self._norm_path(legacy_path):
            return default_path
        return self.text_feat_path

    @property
    def image_feat_ids_path_resolved(self) -> str:
        default_path = str(Path(self.image_feature_dir) / 'item_ids_image.npy')
        legacy_path = str(Path(self.outputs_dir) / 'image_features' / 'item_ids_image.npy')
        if not self.image_feat_ids_path or self._norm_path(self.image_feat_ids_path) == self._norm_path(legacy_path):
            return default_path
        return self.image_feat_ids_path

    @property
    def image_feat_path_resolved(self) -> str:
        default_path = str(Path(self.image_feature_dir) / 'image_features.npy')
        legacy_path = str(Path(self.outputs_dir) / 'image_features' / 'image_features.npy')
        if not self.image_feat_path or self._norm_path(self.image_feat_path) == self._norm_path(legacy_path):
            return default_path
        return self.image_feat_path

    def to_dict(self) -> dict:
        data = asdict(self)
        data['text_feature_dir'] = self.text_feature_dir
        data['image_feature_dir'] = self.image_feature_dir
        data['text_feat_ids_path_resolved'] = self.text_feat_ids_path_resolved
        data['text_feat_path_resolved'] = self.text_feat_path_resolved
        data['image_feat_ids_path_resolved'] = self.image_feat_ids_path_resolved
        data['image_feat_path_resolved'] = self.image_feat_path_resolved
        return data
