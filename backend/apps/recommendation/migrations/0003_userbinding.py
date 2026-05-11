from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):
    dependencies = [
        ('auth', '0012_alter_user_first_name_max_length'),
        ('recommendation', '0002_rename_indexes'),
    ]

    operations = [
        migrations.CreateModel(
            name='UserBinding',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('display_name', models.CharField(blank=True, max_length=128)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('recommendation_user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='bound_accounts', to='recommendation.userprofile')),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, related_name='recommendation_binding', to='auth.user')),
            ],
        ),
    ]
