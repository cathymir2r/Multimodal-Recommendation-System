<template>
  <article class="panel stack-panel">
    <section>
      <div class="panel-head">
        <h2>推荐历史</h2>
        <button class="ghost" @click="$emit('refresh-history')">刷新</button>
      </div>
      <div v-if="history.length" class="history-list">
        <div v-for="row in history" :key="row.id" class="history-row">
          <div>
            <strong>{{ row.account_label || row.user_id }}</strong>
            <span>{{ formatDate(row.created_at) }}</span>
          </div>
          <div>TopK: {{ row.request_top_k }} / 返回 {{ row.item_count }} 条</div>
        </div>
      </div>
      <p v-else class="empty">暂无推荐日志。</p>
    </section>
    <section class="catalog-section">
      <div class="panel-head">
        <h2>商品目录</h2>
      </div>
      <div class="controls slim-controls">
        <input :value="catalogQuery" placeholder="按标题搜索商品" @input="$emit('update:catalogQuery', $event.target.value)" @keyup.enter="$emit('refresh-catalog')" />
        <button @click="$emit('refresh-catalog')">搜索</button>
      </div>
      <div class="catalog-grid">
        <button v-for="item in catalogItems" :key="item.item_id" class="catalog-card" @click="$emit('select-product', item.item_id)">
          <img v-if="item.has_image !== false" :src="item.image_url || placeholderImage" @error="onImageError($event)" />
          <div v-else class="placeholder-cover">
            <span class="placeholder-badge">暂无图片</span>
            <strong>{{ shortTitle(item.title || item.item_id) }}</strong>
            <small>{{ cleanLabel(item.category || '商品') }}</small>
          </div>
          <div>
            <strong>{{ item.title || item.item_id }}</strong>
            <p>{{ cleanLabel(item.brand || '未知品牌') }}</p>
          </div>
        </button>
      </div>
    </section>
  </article>
</template>

<script setup>
import { formatDate, placeholderImage } from '../lib/api'

defineProps({
  history: { type: Array, default: () => [] },
  catalogItems: { type: Array, default: () => [] },
  catalogQuery: { type: String, default: '' },
})

defineEmits(['refresh-history', 'refresh-catalog', 'update:catalogQuery', 'select-product'])

function onImageError(event) {
  event.target.onerror = null
  event.target.alt = ''
  event.target.src = placeholderImage
}

function cleanLabel(value) {
  const text = String(value || '').trim()
  if (!text || text.toLowerCase() === 'nan') {
    return '未知品牌'
  }
  return text
}

function shortTitle(value) {
  const text = String(value || '').trim()
  return text ? text.slice(0, 18) : '商品'
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
.stack-panel {
  display: grid;
  gap: 28px;
}
.panel-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 16px;
}
.panel-head h2 { margin: 0; font-size: 22px; }
.ghost {
  border: 1px solid var(--line);
  background: transparent;
  border-radius: 12px;
  padding: 10px 12px;
  cursor: pointer;
}
.history-list { display: grid; gap: 12px; }
.history-row {
  display: flex;
  justify-content: space-between;
  gap: 16px;
  padding: 14px;
  border: 1px solid var(--line);
  border-radius: 16px;
  background: #fff;
}
.history-row span,
.history-row div:last-child { color: var(--muted); font-size: 13px; }
.controls { display: flex; gap: 12px; margin-bottom: 16px; }
.slim-controls { max-width: 520px; }
.controls input {
  flex: 1;
  padding: 14px;
  border-radius: 14px;
  border: 1px solid var(--line);
}
.controls button {
  border: 0;
  border-radius: 14px;
  padding: 14px 18px;
  background: linear-gradient(135deg, var(--accent), #115e59);
  color: #fff;
  cursor: pointer;
}
.catalog-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 14px;
}
.catalog-card {
  overflow: hidden;
  border-radius: 18px;
  background: #fff;
  border: 1px solid var(--line);
  padding: 0;
  cursor: pointer;
  text-align: left;
}
.catalog-card img,
.placeholder-cover {
  width: 100%;
  height: 220px;
  background: linear-gradient(145deg, #eef2f7, #dfe7ef);
}
.catalog-card img {
  object-fit: contain;
  padding: 12px;
}
.placeholder-cover {
  display: flex;
  flex-direction: column;
  justify-content: end;
  gap: 8px;
  padding: 16px;
}
.placeholder-badge {
  align-self: start;
  padding: 6px 10px;
  border-radius: 999px;
  background: rgba(15,118,110,0.12);
  color: var(--accent);
  font-size: 12px;
}
.placeholder-cover strong {
  font-size: 18px;
  line-height: 1.3;
}
.placeholder-cover small {
  color: var(--muted);
}
.catalog-card div:last-child { padding: 12px; }
.catalog-card strong { display: block; line-height: 1.5; }
.catalog-card p { margin: 6px 0 0; color: var(--muted); font-size: 13px; }
.empty { color: var(--muted); }
@media (max-width: 640px) {
  .controls { flex-direction: column; }
  .history-row { flex-direction: column; }
}
</style>
