from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name='Product',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('item_id', models.CharField(max_length=128, unique=True)),
                ('title', models.CharField(blank=True, max_length=512)),
                ('brand', models.CharField(blank=True, max_length=256)),
                ('category', models.CharField(blank=True, max_length=256)),
                ('image_url', models.TextField(blank=True)),
                ('average_rating', models.FloatField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='UserProfile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('user_id', models.CharField(max_length=128, unique=True)),
                ('last_active_at', models.DateTimeField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='UserBehavior',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('behavior_type', models.CharField(choices=[('view', 'view'), ('click', 'click'), ('favorite', 'favorite'), ('purchase', 'purchase'), ('rate', 'rate')], default='rate', max_length=32)),
                ('score', models.FloatField(blank=True, null=True)),
                ('occurred_at', models.DateTimeField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('product', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='behaviors', to='recommendation.product')),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='behaviors', to='recommendation.userprofile')),
            ],
            options={
                'indexes': [models.Index(fields=['user', 'occurred_at'], name='recommenda_user_id_b39452_idx'), models.Index(fields=['product', 'occurred_at'], name='recommenda_product_b0f00b_idx')],
            },
        ),
        migrations.CreateModel(
            name='RecommendationLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('request_top_k', models.PositiveIntegerField(default=10)),
                ('result_payload', models.JSONField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='recommendation_logs', to='recommendation.userprofile')),
            ],
        ),
    ]
