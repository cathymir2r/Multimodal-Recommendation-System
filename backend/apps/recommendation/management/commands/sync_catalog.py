from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone
import pandas as pd

from apps.recommendation.models import Product, UserBehavior, UserProfile


class Command(BaseCommand):
    help = "Sync users, products and rating behaviors from amazon_clothing.csv into Django models."

    def add_arguments(self, parser):
        parser.add_argument("--csv", dest="csv_path", default=None)
        parser.add_argument("--limit", dest="limit", type=int, default=None)
        parser.add_argument("--clear", dest="clear", action="store_true")

    @transaction.atomic
    def handle(self, *args, **options):
        from django.conf import settings

        csv_path = options["csv_path"] or settings.MULTIMODAL_CONFIG["DATA_CSV_PATH"]
        limit = options["limit"]
        clear = options["clear"]

        df = pd.read_csv(csv_path)
        if limit:
            df = df.head(limit).copy()

        df["user_id"] = df["user_id"].astype(str)
        df["item_id"] = df["item_id"].astype(str)

        if clear:
            UserBehavior.objects.all().delete()
            Product.objects.all().delete()
            UserProfile.objects.all().delete()

        user_ids = sorted(df["user_id"].dropna().unique().tolist())
        item_meta = df.drop_duplicates(subset=["item_id"]).copy()

        existing_users = set(UserProfile.objects.filter(user_id__in=user_ids).values_list("user_id", flat=True))
        UserProfile.objects.bulk_create([
            UserProfile(user_id=user_id) for user_id in user_ids if user_id not in existing_users
        ], batch_size=1000)

        products_to_create = []
        existing_products = set(Product.objects.filter(item_id__in=item_meta["item_id"].tolist()).values_list("item_id", flat=True))
        for _, row in item_meta.iterrows():
            item_id = str(row["item_id"])
            if item_id in existing_products:
                continue
            products_to_create.append(Product(
                item_id=item_id,
                title=str(row.get("title", "") or "")[:512],
                brand=str(row.get("brand", "") or "")[:256],
                category=str(row.get("category", "") or "")[:256],
                image_url=str(row.get("image_url", "") or ""),
                average_rating=float(row["rating"]) if pd.notna(row.get("rating")) else None,
            ))
        Product.objects.bulk_create(products_to_create, batch_size=1000)

        user_map = {obj.user_id: obj for obj in UserProfile.objects.filter(user_id__in=user_ids)}
        product_map = {obj.item_id: obj for obj in Product.objects.filter(item_id__in=item_meta["item_id"].tolist())}

        behaviors = []
        for _, row in df.iterrows():
            user_id = str(row["user_id"])
            item_id = str(row["item_id"])
            if user_id not in user_map or item_id not in product_map:
                continue
            timestamp = timezone.now()
            if pd.notna(row.get("time")):
                timestamp = timezone.datetime.fromtimestamp(int(row["time"]), tz=timezone.get_current_timezone())
            behaviors.append(UserBehavior(
                user=user_map[user_id],
                product=product_map[item_id],
                behavior_type="rate",
                score=float(row["rating"]) if pd.notna(row.get("rating")) else None,
                occurred_at=timestamp,
            ))

        if clear:
            UserBehavior.objects.bulk_create(behaviors, batch_size=1000)
        else:
            UserBehavior.objects.bulk_create(behaviors[:50000], batch_size=1000)

        self.stdout.write(self.style.SUCCESS(
            f"sync complete: users={len(user_ids)}, products={len(product_map)}, behaviors={len(behaviors if clear else behaviors[:50000])}"
        ))
