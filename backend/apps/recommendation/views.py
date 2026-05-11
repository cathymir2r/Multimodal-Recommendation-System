import json
import os
import random

from django.conf import settings
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.models import User
from django.core.cache import cache
from django.core.paginator import Paginator
from django.db.models import Avg, Count, Max
from django.http import FileResponse, HttpResponse, JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from .cache_utils import get_or_set_json, invalidate_prefix
from .models import Product, RecommendationLog, UserBehavior, UserBinding, UserProfile
from .services import recommendation_service




def normalize_image_url(raw_value, fallback_item_id: str | None = None):
    candidates = recommendation_service.parse_image_candidates(raw_value)
    if candidates:
        return candidates[0]
    if fallback_item_id:
        return f'/api/image/{fallback_item_id}/'
    return None


def product_has_image(product) -> bool:
    value = str(getattr(product, 'image_url', '') or '').strip()
    return bool(value and value.lower() != 'nan')




def get_available_recommendation_user(recommendation_user_id: str = ''):
    recommendation_service.initialize()
    available_ids = set(recommendation_service.user2idx.keys())
    if recommendation_user_id:
        if recommendation_user_id not in available_ids:
            return None
        return UserProfile.objects.filter(user_id=recommendation_user_id).first()
    candidates = list(UserProfile.objects.filter(user_id__in=available_ids))
    if not candidates:
        return None
    usage = {
        row['recommendation_user']: row['total']
        for row in UserBinding.objects.values('recommendation_user').annotate(total=Count('id'))
    }
    unbound_candidates = [user for user in candidates if usage.get(user.id, 0) == 0]
    if unbound_candidates:
        return random.choice(unbound_candidates)
    candidates.sort(key=lambda user: (usage.get(user.id, 0), user.user_id))
    lowest_usage = usage.get(candidates[0].id, 0)
    least_used = [user for user in candidates if usage.get(user.id, 0) == lowest_usage]
    return random.choice(least_used)

def ensure_user_binding(user, display_name: str = ''):
    binding = getattr(user, 'recommendation_binding', None)
    if binding is not None:
        updated = False
        if display_name and not binding.display_name:
            binding.display_name = display_name
            updated = True
        if updated:
            binding.save(update_fields=['display_name'])
        return binding

    recommendation_user = get_available_recommendation_user()
    if recommendation_user is None:
        return None
    return UserBinding.objects.create(
        user=user,
        recommendation_user=recommendation_user,
        display_name=display_name,
    )


def build_auth_payload(request):
    if not request.user.is_authenticated:
        return {'is_authenticated': False}
    binding = ensure_user_binding(request.user)
    return {
        'is_authenticated': True,
        'username': request.user.username,
        'display_name': binding.display_name if binding and binding.display_name else request.user.username,
        'recommendation_user_id': binding.recommendation_user.user_id if binding else None,
    }

def load_report(path: str):
    if not os.path.exists(path):
        return None
    with open(path, 'r', encoding='utf-8') as file:
        return json.load(file)


def build_status_payload():
    config = settings.MULTIMODAL_CONFIG
    return {
        'database_engine': settings.DATABASES['default']['ENGINE'],
        'cache_backend': settings.CACHES['default']['BACKEND'],
        'users': UserProfile.objects.count(),
        'products': Product.objects.count(),
        'behaviors': UserBehavior.objects.count(),
        'recommendation_logs': RecommendationLog.objects.count(),
        'artifacts': {
            'model_exists': os.path.exists(config['MODEL_SAVE_PATH']),
            'text_features_exists': os.path.exists(config['TEXT_FEAT_PATH']),
            'image_features_exists': os.path.exists(config['IMAGE_FEAT_PATH']),
            'summary_report_exists': os.path.exists(config['SUMMARY_REPORT_PATH']),
            'insights_report_exists': os.path.exists(config['INSIGHTS_REPORT_PATH']),
        },
    }


def build_architecture_payload():
    return {
        'research_pipeline': [
            {'stage': '数据预处理', 'detail': '对商品标题、描述与评论文本进行清洗，并按时间顺序构建用户行为序列样本。'},
            {'stage': '多模态特征提取', 'detail': '使用 BERT 提取文本语义特征，使用 ResNet 或 ViT 提取商品视觉特征。'},
            {'stage': '特征对齐与融合', 'detail': '通过投影层、门控机制和注意力机制实现文本与图像特征的对齐与融合。'},
            {'stage': '个性化排序推荐', 'detail': '结合用户历史行为序列与目标商品的多模态表示，输出个性化推荐结果。'},
        ],
        'system_modules': [
            {'name': 'Django 后端服务', 'detail': '提供推荐接口、商品详情、用户画像、实验报告、系统状态与图片代理能力。'},
            {'name': 'Vue 前端展示', 'detail': '提供系统总览、实验分析、用户画像、架构说明和商品目录等可视化页面。'},
            {'name': '模型训练与实验模块', 'detail': '负责特征提取、基线实验、主模型训练、消融实验和结果汇总分析。'},
            {'name': '数据存储与缓存模块', 'detail': '使用 SQLite 或 MySQL 持久化数据，结合本地缓存或 Redis 提升查询效率。'},
        ],
        'deployment': [
            {'layer': '前端层', 'tech': 'Vue 3 + Vite'},
            {'layer': '后端层', 'tech': 'Django 5'},
            {'layer': '模型层', 'tech': 'PyTorch'},
            {'layer': '数据库层', 'tech': 'SQLite / MySQL'},
            {'layer': '缓存层', 'tech': 'LocMem / Redis'},
            {'layer': '部署层', 'tech': 'Docker Compose'},
        ],
    }


def build_api_docs_payload():
    return {
        'groups': [
            {
                'name': 'System',
                'endpoints': [
                    {'method': 'GET', 'path': '/api/dashboard/', 'description': 'Return dashboard bootstrap payload.'},
                    {'method': 'GET', 'path': '/api/status/', 'description': 'Return runtime system status and artifact readiness.'},
                    {'method': 'GET', 'path': '/api/architecture/', 'description': 'Return research pipeline and system architecture data.'},
                ],
            },
            {
                'name': 'Experiments',
                'endpoints': [
                    {'method': 'GET', 'path': '/api/experiments/', 'description': 'Return baseline, sequence model, summary, automated insights, and chart metadata.'},
                    {'method': 'GET', 'path': '/api/experiments/figures/<filename>/', 'description': 'Return generated experiment figure files.'},
                    {'method': 'GET', 'path': '/api/export/summary/', 'description': 'Export experiment summary as Markdown-friendly text.'},
                    {'method': 'GET', 'path': '/api/export/appendix/', 'description': 'Export thesis appendix style experiment and system details.'},
                ],
            },
            {
                'name': 'Catalog and Profiles',
                'endpoints': [
                    {'method': 'GET', 'path': '/api/catalog/', 'description': 'Return paginated product catalog with optional search.'},
                    {'method': 'GET', 'path': '/api/products/<item_id>/', 'description': 'Return product detail and recent interactions.'},
                    {'method': 'GET', 'path': '/api/users/<user_id>/', 'description': 'Return user profile and recent behaviors.'},
                ],
            },
            {
                'name': 'Recommendation',
                'endpoints': [
                    {'method': 'GET', 'path': '/api/recommendations/history/', 'description': 'Return recent recommendation logs.'},
                    {'method': 'POST', 'path': '/api/recommend/', 'description': 'Return personalized recommendation items for a user.'},
                    {'method': 'GET', 'path': '/api/image/<item_id>/', 'description': 'Proxy and cache product image.'},
                ],
            },
        ]
    }


def build_export_summary_text():
    config = settings.MULTIMODAL_CONFIG
    summary = load_report(config['SUMMARY_REPORT_PATH']) or {'results': []}
    architecture = build_architecture_payload()
    lines = [
        '# Multi-modal Recommendation System Summary',
        '',
        '## Research Pipeline',
    ]
    for step in architecture['research_pipeline']:
        lines.append(f"- {step['stage']}: {step['detail']}")
    lines.extend(['', '## Experiment Results'])
    for row in summary.get('results', []):
        lines.append(
            f"- {row.get('model_name')}: MSE={row.get('best_val_mse')}, "
            f"F1={row.get('f1')}, Hit@10={row.get('hit@10')}, NDCG@10={row.get('ndcg@10')}"
        )
    insights = load_report(config['INSIGHTS_REPORT_PATH']) or {'highlights': []}
    lines.extend(['', '## Automated Insights'])
    for insight in insights.get('highlights', []):
        lines.append(f"- {insight}")
    lines.extend(['', '## System Stack'])
    for item in architecture['deployment']:
        lines.append(f"- {item['layer']}: {item['tech']}")
    return '\n'.join(lines)


def build_export_appendix_text():
    config = settings.MULTIMODAL_CONFIG
    summary = load_report(config['SUMMARY_REPORT_PATH']) or {'results': []}
    sequence = load_report(config['SEQUENCE_REPORT_PATH']) or {}
    baselines = load_report(config['BASELINE_REPORT_PATH']) or {'results': []}
    insights = load_report(config['INSIGHTS_REPORT_PATH']) or {'highlights': [], 'summary_cards': [], 'charts': []}
    architecture = build_architecture_payload()

    lines = [
        '# Thesis Appendix: Multi-modal Recommendation System',
        '',
        '## A. System Architecture Modules',
    ]
    for module in architecture['system_modules']:
        lines.append(f"- {module['name']}: {module['detail']}")

    lines.extend(['', '## B. Main Model Configuration'])
    main_config = sequence.get('config', {})
    for key in [
        'image_backbone', 'sequence_encoder_type', 'num_attention_heads',
        'num_transformer_layers', 'alignment_weight', 'batch_size',
        'num_epochs', 'learning_rate', 'max_seq_len', 'hidden_dim', 'user_emb_dim'
    ]:
        if key in main_config:
            lines.append(f"- {key}: {main_config.get(key)}")

    lines.extend(['', '## C. Baseline And Main Results'])
    for row in summary.get('results', []):
        lines.append(
            f"- [{row.get('group')}] {row.get('model_name')} | backbone={row.get('image_backbone')} | "
            f"encoder={row.get('sequence_encoder_type')} | MSE={row.get('best_val_mse')} | "
            f"F1={row.get('f1')} | Hit@10={row.get('hit@10')} | NDCG@10={row.get('ndcg@10')}"
        )

    lines.extend(['', '## D. Automated Findings'])
    for insight in insights.get('highlights', []):
        lines.append(f"- {insight}")

    lines.extend(['', '## E. Summary Cards'])
    for card in insights.get('summary_cards', []):
        lines.append(f"- {card.get('title')}: {card.get('value')} | {card.get('detail')}")

    lines.extend(['', '## F. Figure Assets'])
    for chart in insights.get('charts', []):
        lines.append(f"- {chart.get('name')} ({chart.get('metric')}): {chart.get('caption')}")

    lines.extend(['', '## G. Baseline Count'])
    lines.append(f"- number_of_baselines: {len(baselines.get('results', []))}")
    lines.append(f"- number_of_summary_rows: {len(summary.get('results', []))}")

    return '\n'.join(lines)


def build_catalog_payload(query: str = '', page: int = 1, page_size: int = 12):
    queryset = Product.objects.exclude(image_url='').exclude(image_url__isnull=True).order_by('title', 'item_id')
    if query:
        queryset = queryset.filter(title__icontains=query)
    paginator = Paginator(queryset, page_size)
    page_obj = paginator.get_page(page)
    return {
        'page': page_obj.number,
        'page_size': page_size,
        'total_pages': paginator.num_pages,
        'total_items': paginator.count,
        'items': [
            {
                'item_id': product.item_id,
                'title': product.title,
                'brand': product.brand,
                'category': product.category,
                'image_url': f'/api/image/{product.item_id}/',
                'has_image': product_has_image(product),
                'average_rating': product.average_rating,
            }
            for product in page_obj.object_list
        ],
    }


def build_history_payload(page: int = 1, page_size: int = 10):
    queryset = RecommendationLog.objects.select_related('user').order_by('-created_at')
    paginator = Paginator(queryset, page_size)
    page_obj = paginator.get_page(page)
    rows = []
    for log in page_obj.object_list:
        payload = log.result_payload or {}
        account_username = str(payload.get('account_username', '') or '').strip()
        account_display_name = str(payload.get('account_display_name', '') or '').strip()
        account_label = account_display_name or account_username or log.user.user_id
        rows.append({
            'id': log.id,
            'user_id': log.user.user_id,
            'account_username': account_username or None,
            'account_display_name': account_display_name or None,
            'account_label': account_label,
            'request_top_k': log.request_top_k,
            'created_at': log.created_at,
            'item_count': len(payload.get('items', [])),
            'preview_items': payload.get('items', [])[:3],
            'result_items': payload.get('items', []),
        })
    return {
        'page': page_obj.number,
        'page_size': page_size,
        'total_pages': paginator.num_pages,
        'total_items': paginator.count,
        'items': rows,
    }


def build_user_binding_payload():
    rows = []
    bindings = UserBinding.objects.select_related('user', 'recommendation_user').order_by('user__username')
    for binding in bindings:
        rows.append({
            'username': binding.user.username,
            'display_name': binding.display_name or binding.user.username,
            'recommendation_user_id': binding.recommendation_user.user_id,
            'created_at': binding.created_at,
        })
    return {'items': rows, 'total_items': len(rows)}


@require_GET
def index(request):
    return render(request, 'recommendation/index.html')


@require_GET
def dashboard_bootstrap(request):
    config = settings.MULTIMODAL_CONFIG
    static_payload = get_or_set_json(
        'dashboard_bootstrap_static',
        {'page': 1, 'page_size': 8},
        lambda: {
            'status': build_status_payload(),
            'experiments': {
                'summary': load_report(config['SUMMARY_REPORT_PATH']),
                'baselines': load_report(config['BASELINE_REPORT_PATH']),
                'sequence': load_report(config['SEQUENCE_REPORT_PATH']),
            },
            'catalog': build_catalog_payload(page=1, page_size=8),
            'history': build_history_payload(page=1, page_size=8),
            'featured_products': [
                {
                    'item_id': product.item_id,
                    'title': product.title,
                    'brand': product.brand,
                    'category': product.category,
                    'image_url': f'/api/image/{product.item_id}/',
                    'has_image': product_has_image(product),
                    'average_rating': product.average_rating,
                }
                for product in Product.objects.exclude(image_url='').exclude(image_url__isnull=True).order_by('-average_rating', 'title')[:6]
            ],
        },
        timeout=120,
    )
    payload = {**static_payload, 'auth': build_auth_payload(request)}
    return JsonResponse(payload)


@require_GET
def architecture_info(request):
    payload = get_or_set_json('architecture_info', {'version': 1}, build_architecture_payload, timeout=600)
    return JsonResponse(payload)


@require_GET
def api_docs(request):
    payload = get_or_set_json('api_docs', {'version': 1}, build_api_docs_payload, timeout=600)
    return JsonResponse(payload)


@require_GET
def export_summary(request):
    text = build_export_summary_text()
    return HttpResponse(text, content_type='text/plain; charset=utf-8')


@require_GET
def export_appendix(request):
    text = build_export_appendix_text()
    return HttpResponse(text, content_type='text/plain; charset=utf-8')


@require_GET
def system_status(request):
    cache_key = 'system_status_payload'
    cached_payload = cache.get(cache_key)
    if cached_payload is not None:
        return JsonResponse(cached_payload)
    payload = build_status_payload()
    cache.set(cache_key, payload, timeout=60)
    return JsonResponse(payload)


def _experiment_charts_payload():
    config = settings.MULTIMODAL_CONFIG
    insights = load_report(config['INSIGHTS_REPORT_PATH']) or {'highlights': [], 'summary_cards': [], 'charts': []}
    charts = []
    for chart in insights.get('charts', []):
        charts.append({
            'name': chart.get('name'),
            'metric': chart.get('metric'),
            'url': f"/api/experiments/figures/{chart.get('name')}",
            'caption': chart.get('caption'),
        })
    return {
        'highlights': insights.get('highlights', []),
        'summary_cards': insights.get('summary_cards', []),
        'charts': charts,
    }


@require_GET
def experiment_figure(request, filename: str):
    safe_name = os.path.basename(filename)
    figure_path = os.path.join(settings.MULTIMODAL_CONFIG['REPORT_FIGURES_DIR'], safe_name)
    if not os.path.exists(figure_path):
        return JsonResponse({'detail': 'figure not found'}, status=404)
    return FileResponse(open(figure_path, 'rb'), content_type='image/png')


@require_GET
def experiment_results(request):
    config = settings.MULTIMODAL_CONFIG
    summary = load_report(config['SUMMARY_REPORT_PATH'])
    baselines = load_report(config['BASELINE_REPORT_PATH'])
    sequence = load_report(config['SEQUENCE_REPORT_PATH'])
    insights = load_report(config['INSIGHTS_REPORT_PATH'])
    charts = _experiment_charts_payload()
    return JsonResponse({'summary': summary, 'baselines': baselines, 'sequence': sequence, 'insights': insights, 'charts': charts})


@require_GET
def catalog_list(request):
    page = max(int(request.GET.get('page', 1)), 1)
    page_size = min(max(int(request.GET.get('page_size', 12)), 1), 50)
    query = str(request.GET.get('q', '')).strip()
    payload = get_or_set_json(
        'catalog_list',
        {'query': query, 'page': page, 'page_size': page_size},
        lambda: build_catalog_payload(query=query, page=page, page_size=page_size),
        timeout=120,
    )
    return JsonResponse(payload)


@require_GET
def product_detail(request, item_id: str):
    cache_payload = get_or_set_json(
        'product_detail',
        {'item_id': str(item_id)},
        lambda: _product_detail_payload(item_id),
        timeout=300,
    )
    status = 404 if cache_payload.get('detail') == 'product not found' else 200
    return JsonResponse(cache_payload, status=status)


def _product_detail_payload(item_id: str):
    try:
        product = Product.objects.get(item_id=str(item_id))
    except Product.DoesNotExist:
        return {'detail': 'product not found'}

    behavior_summary = product.behaviors.aggregate(
        behavior_count=Count('id'),
        mean_score=Avg('score'),
        last_behavior_at=Max('occurred_at'),
    )
    recent_behaviors = list(product.behaviors.select_related('user').order_by('-occurred_at')[:10].values(
        'user__user_id', 'behavior_type', 'score', 'occurred_at'
    ))
    return {
        'item_id': product.item_id,
        'title': product.title,
        'brand': product.brand,
        'category': product.category,
        'image_url': f'/api/image/{product.item_id}/',
        'has_image': product_has_image(product),
        'average_rating': product.average_rating,
        'behavior_summary': behavior_summary,
        'recent_behaviors': recent_behaviors,
    }


@require_GET
def user_profile(request, user_id: str):
    payload = get_or_set_json(
        'user_profile',
        {'user_id': str(user_id)},
        lambda: _user_profile_payload(user_id),
        timeout=300,
    )
    status = 404 if payload.get('detail') == 'user not found' else 200
    return JsonResponse(payload, status=status)


def _user_profile_payload(user_id: str):
    try:
        user = UserProfile.objects.get(user_id=str(user_id))
    except UserProfile.DoesNotExist:
        return {'detail': 'user not found'}

    behavior_qs = user.behaviors.select_related('product').order_by('-occurred_at')
    behavior_summary = behavior_qs.aggregate(
        behavior_count=Count('id'),
        mean_score=Avg('score'),
        last_behavior_at=Max('occurred_at'),
    )
    recent_behaviors = [
        {
            'item_id': behavior.product.item_id,
            'title': behavior.product.title,
            'behavior_type': behavior.behavior_type,
            'score': behavior.score,
            'occurred_at': behavior.occurred_at,
        }
        for behavior in behavior_qs[:10]
    ]
    return {
        'user_id': user.user_id,
        'last_active_at': user.last_active_at,
        'created_at': user.created_at,
        'behavior_summary': behavior_summary,
        'recent_behaviors': recent_behaviors,
    }


def serialize_recommendation_item(item):
    return {
        'item_id': item.item_id,
        'score': item.score,
        'title': item.title,
        'rating': item.rating,
        'image_url': f'/api/image/{item.item_id}/' if item.has_image else None,
        'has_image': item.has_image,
    }


@require_GET
def recommendation_history(request):
    page = max(int(request.GET.get('page', 1)), 1)
    page_size = min(max(int(request.GET.get('page_size', 10)), 1), 50)
    payload = get_or_set_json(
        'recommendation_history',
        {'page': page, 'page_size': page_size},
        lambda: build_history_payload(page=page, page_size=page_size),
        timeout=60,
    )
    return JsonResponse(payload)


@require_GET
def user_binding_list(request):
    payload = get_or_set_json('user_binding_list', {'version': 1}, build_user_binding_payload, timeout=120)
    return JsonResponse(payload)


@csrf_exempt
@require_POST
def recommend(request):
    try:
        payload = json.loads(request.body.decode('utf-8')) if request.body else {}
    except json.JSONDecodeError:
        return JsonResponse({'detail': 'invalid json body'}, status=400)

    user_id = str(payload.get('user_id', '')).strip()
    top_k = int(payload.get('top_k', 10))
    if not user_id and request.user.is_authenticated:
        binding = getattr(request.user, 'recommendation_binding', None)
        if binding:
            user_id = binding.recommendation_user.user_id
    if not user_id:
        return JsonResponse({'detail': 'user_id is required'}, status=400)

    cache_payload = {'user_id': user_id, 'top_k': top_k}
    try:
        response_payload = get_or_set_json(
            'recommend_result',
            cache_payload,
            lambda: {
                'user_id': user_id,
                'top_k': top_k,
                'items': [serialize_recommendation_item(item) for item in recommendation_service.recommend(user_id, top_k)],
            },
            timeout=180,
        )
    except ValueError as exc:
        return JsonResponse({'detail': str(exc)}, status=400)
    except Exception as exc:
        return JsonResponse({'detail': f'service error: {exc}'}, status=500)

    user_obj, _ = UserProfile.objects.get_or_create(user_id=user_id)
    log_payload = dict(response_payload)
    if request.user.is_authenticated:
        binding = getattr(request.user, 'recommendation_binding', None)
        log_payload['account_username'] = request.user.username
        log_payload['account_display_name'] = binding.display_name if binding and binding.display_name else request.user.username
    RecommendationLog.objects.create(user=user_obj, request_top_k=top_k, result_payload=log_payload)
    cache.delete('system_status_payload')
    invalidate_prefix('dashboard_bootstrap', [{'page': 1, 'page_size': 8}])
    invalidate_prefix('recommendation_history', [{'page': 1, 'page_size': 8}, {'page': 1, 'page_size': 10}])
    invalidate_prefix('user_profile', [{'user_id': user_id}])
    return JsonResponse(response_payload)


@require_GET
def image_proxy(request, item_id: str):
    try:
        cache_path, media_type = recommendation_service.fetch_image(str(item_id))
    except Exception:
        cache_path, media_type = None, 'image/svg+xml'

    if cache_path:
        return FileResponse(open(cache_path, 'rb'), content_type=media_type)

    svg = "<svg xmlns='http://www.w3.org/2000/svg' width='300' height='300'><rect width='100%25' height='100%25' fill='%23ddd'/><text x='50%25' y='50%25' dominant-baseline='middle' text-anchor='middle' fill='%23666' font-size='18'>No Image</text></svg>"
    return HttpResponse(svg, content_type='image/svg+xml')


@csrf_exempt
@require_POST
def register_account(request):
    try:
        payload = json.loads(request.body.decode('utf-8')) if request.body else {}
    except json.JSONDecodeError:
        return JsonResponse({'detail': 'invalid json body'}, status=400)

    username = str(payload.get('username', '')).strip()
    password = str(payload.get('password', '')).strip()
    display_name = str(payload.get('display_name', '')).strip()
    recommendation_user_id = str(payload.get('recommendation_user_id', '')).strip()

    if len(username) < 3 or len(password) < 6:
        return JsonResponse({'detail': 'username must be >= 3 chars and password >= 6 chars'}, status=400)
    if User.objects.filter(username=username).exists():
        return JsonResponse({'detail': 'username already exists'}, status=400)

    recommendation_user = get_available_recommendation_user(recommendation_user_id)
    if recommendation_user is None:
        if recommendation_user_id:
            return JsonResponse({'detail': 'recommendation_user_id not found in trained recommender users'}, status=400)
        return JsonResponse({'detail': 'no recommendation users available'}, status=400)

    user = User.objects.create_user(username=username, password=password)
    UserBinding.objects.create(
        user=user,
        recommendation_user=recommendation_user,
        display_name=display_name,
    )
    login(request, user)
    return JsonResponse(build_auth_payload(request), status=201)


@csrf_exempt
@require_POST
def login_account(request):
    try:
        payload = json.loads(request.body.decode('utf-8')) if request.body else {}
    except json.JSONDecodeError:
        return JsonResponse({'detail': 'invalid json body'}, status=400)

    username = str(payload.get('username', '')).strip()
    password = str(payload.get('password', '')).strip()
    user = authenticate(request, username=username, password=password)
    if user is None:
        return JsonResponse({'detail': 'invalid username or password'}, status=400)
    ensure_user_binding(user)
    login(request, user)
    return JsonResponse(build_auth_payload(request))


@csrf_exempt
@require_POST
def logout_account(request):
    logout(request)
    return JsonResponse({'is_authenticated': False})


@require_GET
def auth_me(request):
    return JsonResponse(build_auth_payload(request))
