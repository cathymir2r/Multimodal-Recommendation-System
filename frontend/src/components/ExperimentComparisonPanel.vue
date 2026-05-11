<template>
  <article class="panel experiments-panel">
    <div class="panel-head">
      <h2>实验对比</h2>
      <button class="ghost" @click="$emit('refresh')">刷新</button>
    </div>
    <div v-if="rows.length" class="chart-list">
      <div v-for="row in rows" :key="row.model_name" class="chart-card">
        <div class="chart-title-row">
          <strong>{{ translateModelName(row.model_name) }}</strong>
        </div>
        <div class="metric-bar-group">
          <div class="metric-line">
            <label>Hit@10</label>
            <div class="bar-track"><div class="bar-fill warm" :style="barStyle(row['hit@10'], 1)" /></div>
            <span>{{ formatMetric(row['hit@10']) }}</span>
          </div>
          <div class="metric-line">
            <label>NDCG@10</label>
            <div class="bar-track"><div class="bar-fill cool" :style="barStyle(row['ndcg@10'], 1)" /></div>
            <span>{{ formatMetric(row['ndcg@10']) }}</span>
          </div>
        </div>
      </div>
    </div>
    <table v-if="rows.length" class="result-table">
      <thead>
        <tr>
          <th>模型</th>
          <th>命中率 Hit@10</th>
          <th>NDCG@10</th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="row in rows" :key="row.model_name + '-table'">
          <td>{{ translateModelName(row.model_name) }}</td>
          <td>{{ formatMetric(row['hit@10']) }}</td>
          <td>{{ formatMetric(row['ndcg@10']) }}</td>
        </tr>
      </tbody>
    </table>
    <p v-else class="empty">暂未检测到实验汇总文件。请先运行训练与对比脚本。</p>
  </article>
</template>

<script setup>
import { computed } from 'vue'
import { formatMetric } from '../lib/api'

const props = defineProps({
  experiments: { type: Object, default: null },
})

defineEmits(['refresh'])

const rows = computed(() => props.experiments?.summary?.results || [])

function barStyle(value, maxValue) {
  const numeric = Number(value || 0)
  const ratio = Math.max(0, Math.min(100, (numeric / maxValue) * 100))
  return { width: `${ratio}%` }
}

function translateModelName(value) {
  const text = String(value || '')
  return text
    .replace('text_only', '仅文本模型')
    .replace('image_only', '仅图像模型')
    .replace('early_fusion', '早期融合模型')
    .replace('sequence_attention_recommender', '序列注意力主模型')
    .replace('main_model', '主模型')
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
.chart-list {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
  gap: 14px;
  margin-bottom: 18px;
}
.chart-card {
  border: 1px solid var(--line);
  border-radius: 18px;
  background: #fff;
  padding: 14px;
}
.chart-title-row {
  display: flex;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 12px;
  font-size: 13px;
}
.metric-bar-group {
  display: grid;
  gap: 10px;
}
.metric-line {
  display: grid;
  grid-template-columns: 62px 1fr 54px;
  align-items: center;
  gap: 10px;
  font-size: 12px;
  color: var(--muted);
}
.bar-track {
  height: 10px;
  border-radius: 999px;
  background: #edf0f5;
  overflow: hidden;
}
.bar-fill {
  height: 100%;
  border-radius: inherit;
}
.bar-fill.warm { background: linear-gradient(90deg, #b45309, #f59e0b); }
.bar-fill.cool { background: linear-gradient(90deg, #1d4ed8, #60a5fa); }
.result-table {
  width: 100%;
  border-collapse: collapse;
}
.result-table th,
.result-table td {
  padding: 12px;
  border-bottom: 1px solid var(--line);
  text-align: left;
  font-size: 14px;
}
.empty { color: var(--muted); }
</style>
