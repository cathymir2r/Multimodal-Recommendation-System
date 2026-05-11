from django.urls import path

from . import views


urlpatterns = [
    path('', views.index, name='index'),
    path('api/dashboard/', views.dashboard_bootstrap, name='dashboard_bootstrap'),
    path('api/architecture/', views.architecture_info, name='architecture_info'),
    path('api/docs/', views.api_docs, name='api_docs'),
    path('api/export/summary/', views.export_summary, name='export_summary'),
    path('api/export/appendix/', views.export_appendix, name='export_appendix'),
    path('api/status/', views.system_status, name='system_status'),
    path('api/auth/me/', views.auth_me, name='auth_me'),
    path('api/auth/register/', views.register_account, name='register_account'),
    path('api/auth/login/', views.login_account, name='login_account'),
    path('api/auth/logout/', views.logout_account, name='logout_account'),
    path('api/experiments/', views.experiment_results, name='experiment_results'),
    path('api/experiments/figures/<str:filename>/', views.experiment_figure, name='experiment_figure'),
    path('api/catalog/', views.catalog_list, name='catalog_list'),
    path('api/products/<str:item_id>/', views.product_detail, name='product_detail'),
    path('api/users/<str:user_id>/', views.user_profile, name='user_profile'),
    path('api/recommendations/history/', views.recommendation_history, name='recommendation_history'),
    path('api/user-bindings/', views.user_binding_list, name='user_binding_list'),
    path('api/recommend/', views.recommend, name='recommend'),
    path('api/image/<str:item_id>/', views.image_proxy, name='image_proxy'),
]
