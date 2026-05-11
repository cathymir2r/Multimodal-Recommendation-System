from django.contrib import admin

from .models import Product, RecommendationLog, UserBehavior, UserProfile


admin.site.register(UserProfile)
admin.site.register(Product)
admin.site.register(UserBehavior)
admin.site.register(RecommendationLog)
