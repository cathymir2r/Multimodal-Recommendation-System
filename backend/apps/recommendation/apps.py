from django.apps import AppConfig


class RecommendationConfig(AppConfig):
    default_auto_field = "django.db.models.BigAutoField"
    name = "apps.recommendation"
    verbose_name = "多模态推荐"
