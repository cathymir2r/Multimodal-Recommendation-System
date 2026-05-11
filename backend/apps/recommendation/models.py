from django.contrib.auth.models import User
from django.db import models


class UserProfile(models.Model):
    user_id = models.CharField(max_length=128, unique=True)
    last_active_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return self.user_id


class Product(models.Model):
    item_id = models.CharField(max_length=128, unique=True)
    title = models.CharField(max_length=512, blank=True)
    brand = models.CharField(max_length=256, blank=True)
    category = models.CharField(max_length=256, blank=True)
    image_url = models.TextField(blank=True)
    average_rating = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return self.item_id


class UserBehavior(models.Model):
    BEHAVIOR_CHOICES = [
        ("view", "view"),
        ("click", "click"),
        ("favorite", "favorite"),
        ("purchase", "purchase"),
        ("rate", "rate"),
    ]

    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name="behaviors")
    product = models.ForeignKey(Product, on_delete=models.CASCADE, related_name="behaviors")
    behavior_type = models.CharField(max_length=32, choices=BEHAVIOR_CHOICES, default="rate")
    score = models.FloatField(null=True, blank=True)
    occurred_at = models.DateTimeField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        indexes = [
            models.Index(fields=["user", "occurred_at"]),
            models.Index(fields=["product", "occurred_at"]),
        ]


class RecommendationLog(models.Model):
    user = models.ForeignKey(UserProfile, on_delete=models.CASCADE, related_name="recommendation_logs")
    request_top_k = models.PositiveIntegerField(default=10)
    result_payload = models.JSONField()
    created_at = models.DateTimeField(auto_now_add=True)


class UserBinding(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name="recommendation_binding")
    recommendation_user = models.ForeignKey(
        UserProfile,
        on_delete=models.CASCADE,
        related_name="bound_accounts",
    )
    display_name = models.CharField(max_length=128, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self) -> str:
        return f"{self.user.username}->{self.recommendation_user.user_id}"
