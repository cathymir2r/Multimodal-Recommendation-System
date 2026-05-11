from pathlib import Path
import os

from config.utils.env_loader import load_env_file


BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent
load_env_file(BASE_DIR / '.env')

SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", "multimodal-recommendation-demo-secret-key")
DEBUG = os.getenv("DJANGO_DEBUG", "true").lower() == "true"
ALLOWED_HOSTS = [host.strip() for host in os.getenv("DJANGO_ALLOWED_HOSTS", "*").split(",") if host.strip()]

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "apps.recommendation",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [PROJECT_ROOT / "backend" / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"
ASGI_APPLICATION = "config.asgi.application"

DB_ENGINE = os.getenv("DB_ENGINE", "sqlite").lower()
if DB_ENGINE == "mysql":
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.mysql",
            "NAME": os.getenv("MYSQL_DATABASE", "multimodal_rec"),
            "USER": os.getenv("MYSQL_USER", "root"),
            "PASSWORD": os.getenv("MYSQL_PASSWORD", "root"),
            "HOST": os.getenv("MYSQL_HOST", "127.0.0.1"),
            "PORT": os.getenv("MYSQL_PORT", "3306"),
            "OPTIONS": {"charset": "utf8mb4"},
        }
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
        }
    }

REDIS_URL = os.getenv("REDIS_URL", "redis://127.0.0.1:6379/1")
if os.getenv("ENABLE_REDIS_CACHE", "false").lower() == "true":
    CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.redis.RedisCache",
            "LOCATION": REDIS_URL,
            "TIMEOUT": 300,
        }
    }
else:
    CACHES = {
        "default": {
            "BACKEND": "django.core.cache.backends.locmem.LocMemCache",
            "LOCATION": "multimodal-rec-cache",
            "TIMEOUT": 300,
        }
    }

LANGUAGE_CODE = "zh-hans"
TIME_ZONE = "Asia/Shanghai"
USE_I18N = True
USE_TZ = True

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = [PROJECT_ROOT / "backend" / "static"]

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

MULTIMODAL_CONFIG = {
    "DATA_CSV_PATH": str(PROJECT_ROOT / "outputs" / "subsets" / "amazon_clothing_seq_clean_u5_i3_unique_ui_nonzeroimg.csv"),
    "META_JSON_PATH": str(PROJECT_ROOT / "meta_Clothing_Shoes_and_Jewelry.json"),
    "TEXT_FEAT_IDS_PATH": str(PROJECT_ROOT / "outputs" / "text_features" / "item_ids_text.npy"),
    "TEXT_FEAT_PATH": str(PROJECT_ROOT / "outputs" / "text_features" / "text_features.npy"),
    "IMAGE_FEAT_IDS_PATH": str(PROJECT_ROOT / "outputs" / "image_features_resnet50" / "item_ids_image.npy"),
    "IMAGE_FEAT_PATH": str(PROJECT_ROOT / "outputs" / "image_features_resnet50" / "image_features.npy"),
    "MODEL_SAVE_PATH": str(PROJECT_ROOT / "outputs" / "models_seq_clean_u5_i3_unique_ui_nonzeroimg_main_fullrank_bpr30_gru_lr3e4_seed202" / "sequence_attention_recommender.pt"),
    "IMAGE_CACHE_DIR": str(PROJECT_ROOT / "outputs" / "image_cache"),
    "BASELINE_REPORT_PATH": str(PROJECT_ROOT / "outputs" / "reports_seq_clean_u5_i3_unique_ui_nonzeroimg_baselines_fullrank_bpr20" / "baseline_comparison_report.json"),
    "SEQUENCE_REPORT_PATH": str(PROJECT_ROOT / "outputs" / "reports_seq_clean_u5_i3_unique_ui_nonzeroimg_main_fullrank_bpr30_gru_lr3e4_seed202" / "sequence_attention_report.json"),
    "SUMMARY_REPORT_PATH": str(PROJECT_ROOT / "outputs" / "reports_seq_clean_u5_i3_unique_ui_nonzeroimg_web" / "experiment_summary.json"),
    "INSIGHTS_REPORT_PATH": str(PROJECT_ROOT / "outputs" / "reports_seq_clean_u5_i3_unique_ui_nonzeroimg_web" / "experiment_insights.json"),
    "REPORT_FIGURES_DIR": str(PROJECT_ROOT / "outputs" / "reports_seq_clean_u5_i3_unique_ui_nonzeroimg_web" / "figures"),
}


