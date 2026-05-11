<template>
  <article class="panel detail-panel">
    <div class="panel-head">
      <h2>商品详情</h2>
    </div>
    <div v-if="selectedProduct" class="detail-wrap">
      <img v-if="selectedProduct.has_image !== false" class="detail-image" :src="selectedProduct.image_url || placeholderImage" @error="onImageError" />
      <div v-else class="detail-placeholder">
        <span>暂无商品图</span>
        <strong>{{ selectedProduct.title || selectedProduct.item_id }}</strong>
      </div>
      <div>
        <h3 class="detail-title">{{ selectedProduct.title || selectedProduct.item_id }}</h3>
        <p>商品ID: {{ selectedProduct.item_id }}</p>
        <p>品牌: {{ cleanLabel(selectedProduct.brand, '未知品牌') }}</p>
        <p>类目: {{ cleanLabel(selectedProduct.category, '未标注类目') }}</p>
        <p>平均评分: {{ formatMetric(selectedProduct.average_rating) }}</p>
        <p>交互数量: {{ selectedProduct.behavior_summary?.behavior_count ?? 0 }}</p>
        <p>最近交互: {{ formatDate(selectedProduct.behavior_summary?.last_behavior_at) }}</p>
      </div>
    </div>
    <p v-else class="empty">点击推荐结果或商品目录中的卡片查看商品详情。</p>
  </article>
</template>

<script setup>
import { formatDate, formatMetric, placeholderImage } from '../lib/api'

defineProps({
  selectedProduct: { type: Object, default: null },
})

function onImageError(event) {
  event.target.onerror = null
  event.target.alt = ''
  event.target.src = placeholderImage
}

function cleanLabel(value, fallback) {
  const text = String(value || '').trim()
  if (!text || text.toLowerCase() === 'nan') {
    return fallback
  }
  return text
}
</script>

<style scoped>
.panel {
  background: rgba(255,255,255,0.88);
  border: 1px solid var(--line);
  border-radius: 24px;
  padding: 20px;
  box-shadow: 0 18px 40px rgba(17,24,39,0.06);
}
.panel-head { margin-bottom: 16px; }
.panel-head h2 { margin: 0; font-size: 22px; }
.detail-wrap {
  display: grid;
  grid-template-columns: 220px 1fr;
  gap: 16px;
  align-items: start;
}
.detail-image,
.detail-placeholder {
  width: 100%;
  height: 220px;
  background: linear-gradient(145deg, #eef2f7, #dfe7ef);
  border-radius: 18px;
}
.detail-image {
  object-fit: contain;
  padding: 12px;
}
.detail-placeholder {
  display: flex;
  flex-direction: column;
  justify-content: end;
  gap: 10px;
  padding: 16px;
}
.detail-placeholder span {
  align-self: start;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(15,118,110,0.12);
  color: var(--accent);
  font-size: 12px;
}
.detail-placeholder strong {
  line-height: 1.5;
}
.detail-title {
  margin: 0 0 8px;
  font-size: 16px;
  line-height: 1.5;
}
.detail-wrap p {
  margin: 8px 0;
  color: var(--muted);
}
.empty { color: var(--muted); }
@media (max-width: 820px) {
  .detail-wrap { grid-template-columns: 1fr; }
}
</style>
