from collections import defaultdict

from django.core.management.base import BaseCommand
from django.db import transaction

from apps.recommendation.models import UserBinding, UserProfile
from apps.recommendation.services import recommendation_service


class Command(BaseCommand):
    help = 'Rebalance existing account bindings across available recommendation users.'

    def add_arguments(self, parser):
        parser.add_argument(
            '--dry-run',
            action='store_true',
            help='Preview the rebalance result without saving changes.',
        )

    def handle(self, *args, **options):
        dry_run = options['dry_run']
        recommendation_service.initialize()
        available_ids = set(recommendation_service.user2idx.keys())
        candidate_profiles = list(UserProfile.objects.filter(user_id__in=available_ids).order_by('user_id'))
        bindings = list(UserBinding.objects.select_related('user', 'recommendation_user').order_by('created_at', 'id'))

        if not candidate_profiles:
            self.stdout.write(self.style.WARNING('No available recommendation users found.'))
            return
        if not bindings:
            self.stdout.write(self.style.WARNING('No bindings found.'))
            return

        slots = defaultdict(list)
        for profile in candidate_profiles:
            slots[0].append(profile)

        assignment = {}
        usage = defaultdict(int)

        for binding in bindings:
            level = 0
            while not slots[level]:
                level += 1
            profile = slots[level].pop(0)
            assignment[binding.id] = profile
            usage[profile.id] += 1
            slots[level + 1].append(profile)

        changed = 0
        preview_rows = []
        for binding in bindings:
            new_profile = assignment[binding.id]
            if binding.recommendation_user_id != new_profile.id:
                changed += 1
            preview_rows.append((binding.user.username, binding.recommendation_user.user_id, new_profile.user_id))

        self.stdout.write(f'Total bindings: {len(bindings)}')
        self.stdout.write(f'Available recommendation users: {len(candidate_profiles)}')
        self.stdout.write(f'Bindings to update: {changed}')

        for username, old_user_id, new_user_id in preview_rows[:20]:
            marker = 'KEEP' if old_user_id == new_user_id else 'MOVE'
            self.stdout.write(f'[{marker}] {username}: {old_user_id} -> {new_user_id}')
        if len(preview_rows) > 20:
            self.stdout.write('...')

        if dry_run:
            self.stdout.write(self.style.WARNING('Dry run only. No changes were saved.'))
            return

        with transaction.atomic():
            for binding in bindings:
                new_profile = assignment[binding.id]
                if binding.recommendation_user_id != new_profile.id:
                    binding.recommendation_user = new_profile
                    binding.save(update_fields=['recommendation_user'])

        self.stdout.write(self.style.SUCCESS(f'Rebalanced {changed} bindings successfully.'))
