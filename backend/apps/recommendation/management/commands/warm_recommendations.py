from django.core.management.base import BaseCommand
from django.test import Client


class Command(BaseCommand):
    help = "Warm up dashboard, catalog and recommend caches for common entry points."

    def add_arguments(self, parser):
        parser.add_argument('--user-id', dest='user_id', default=None)
        parser.add_argument('--top-k', dest='top_k', type=int, default=10)

    def handle(self, *args, **options):
        client = Client()
        warmed = []

        if client.get('/api/dashboard/').status_code == 200:
            warmed.append('dashboard')
        if client.get('/api/status/').status_code == 200:
            warmed.append('status')
        if client.get('/api/catalog/?page_size=8').status_code == 200:
            warmed.append('catalog')
        if client.get('/api/recommendations/history/?page_size=8').status_code == 200:
            warmed.append('history')

        user_id = options.get('user_id')
        if user_id:
            response = client.post(
                '/api/recommend/',
                data=f'{{"user_id":"{user_id}","top_k":{options["top_k"]}}}',
                content_type='application/json',
            )
            if response.status_code == 200:
                warmed.append(f'recommend:{user_id}')

        self.stdout.write(self.style.SUCCESS(f'warmed caches: {", ".join(warmed) if warmed else "none"}'))
