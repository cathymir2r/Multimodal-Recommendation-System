from django.db import migrations


class Migration(migrations.Migration):
    dependencies = [
        ('recommendation', '0001_initial'),
    ]

    operations = [
        migrations.RenameIndex(
            model_name='userbehavior',
            new_name='recommendat_user_id_4d9a2d_idx',
            old_name='recommenda_user_id_b39452_idx',
        ),
        migrations.RenameIndex(
            model_name='userbehavior',
            new_name='recommendat_product_56540a_idx',
            old_name='recommenda_product_b0f00b_idx',
        ),
    ]
