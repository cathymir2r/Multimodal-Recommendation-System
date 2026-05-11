<template>
  <section class="experiment-layout">
    <article class="panel briefing-panel">
      <div class="panel-head">
        <h2>汇报摘要</h2>
      </div>
      <div class="briefing-grid">
        <div class="brief-card">
          <span>&#30740;&#31350;&#30446;&#26631;</span>
          <strong>&#22810;&#27169;&#24577;&#34701;&#21512;&#26041;&#27861;&#30340;&#25928;&#26524;&#23545;&#27604;</strong>
          <p>&#23454;&#39564;&#32467;&#26524;&#23637;&#31034;&#20102;&#25991;&#26412;&#29305;&#24449;&#12289;&#22270;&#20687;&#29305;&#24449;&#12289;&#34892;&#20026;&#24207;&#21015;&#24314;&#27169;&#19982;&#34701;&#21512;&#31574;&#30053;&#22312;&#25512;&#33616;&#24615;&#33021;&#19978;&#30340;&#24046;&#24322;&#12290;</p>
        </div>
        <div class="brief-card">
          <span>&#23454;&#39564;&#32467;&#35770;</span>
          <strong>&#20027;&#27169;&#22411;&#19982;&#22522;&#32447;&#27169;&#22411;&#23384;&#22312;&#24615;&#33021;&#24046;&#24322;</strong>
          <p>&#23454;&#39564;&#32467;&#26524;&#21453;&#26144;&#20102;&#27880;&#24847;&#21147;&#26426;&#21046;&#12289;&#35270;&#35273;&#39592;&#24178;&#32593;&#32476;&#19982;&#24207;&#21015;&#32534;&#30721;&#26041;&#24335;&#23545;&#25512;&#33616;&#25928;&#26524;&#30340;&#19981;&#21516;&#24433;&#21709;&#12290;</p>
        </div>
        <div class="brief-card">
          <span>&#35780;&#20215;&#25351;&#26631;</span>
          <strong>&#25490;&#24207;&#25351;&#26631;&#23545;&#27604;&#35780;&#20215;</strong>
          <p>&#35780;&#20215;&#20307;&#31995;&#37319;&#29992; Hit@10 &#19982; NDCG@10 &#20004;&#20010;&#25490;&#24207;&#25351;&#26631;&#65292;&#29992;&#20110;&#34913;&#37327; Top-K &#25512;&#33616;&#25928;&#26524;&#12290;</p>
        </div>
      </div>
    </article>

    <article class="panel controls-panel">
      <div class="panel-head">
        <h2>实验筛选</h2>
      </div>
      <div class="control-grid">
        <label>
          <span>实验分组</span>
          <select v-model="selectedGroup">
            <option value="all">全部</option>
            <option v-for="group in availableGroups" :key="group" :value="group">{{ translateGroup(group) }}</option>
          </select>
        </label>
        <label>
          <span>视觉骨干</span>
          <select v-model="selectedBackbone">
            <option value="all">全部</option>
            <option v-for="backbone in availableBackbones" :key="backbone" :value="backbone">{{ translateBackbone(backbone) }}</option>
          </select>
        </label>
        <label>
          <span>序列编码</span>
          <select v-model="selectedEncoder">
            <option value="all">全部</option>
            <option v-for="encoder in availableEncoders" :key="encoder" :value="encoder">{{ translateEncoder(encoder) }}</option>
          </select>
        </label>
        <label>
          <span>排序指标</span>
          <select v-model="sortMetric">
            <option value="hit@10">命中率 Hit@10</option>
            <option value="ndcg@10">NDCG@10</option>
          </select>
        </label>
      </div>
      <p class="hint">当前共筛出 {{ filteredRows.length }} 个实验结果。</p>
    </article>

    <article class="panel card-panel">
      <div class="panel-head">
        <h2>研究结论总览</h2>
      </div>
      <div v-if="summaryCards.length" class="summary-grid">
        <div v-for="card in summaryCards" :key="card.title" class="summary-card">
          <span>{{ translateSummaryTitle(card.title) }}</span>
          <strong>{{ translateCardValue(card.title, card.value) }}</strong>
          <p>{{ translateSummaryDetail(card.detail) }}</p>
        </div>
      </div>
      <p v-else class="empty">暂无结论总览卡片，请先生成实验汇总。</p>
    </article>

    <ExperimentComparisonPanel :experiments="filteredExperimentsPayload" @refresh="loadData" />

    <article class="panel ranking-panel">
      <div class="panel-head">
        <h2>Top 排行</h2>
      </div>
      <div v-if="topRows.length" class="ranking-list">
        <div v-for="(row, index) in topRows" :key="`${row.model_name}-${index}`" class="ranking-card">
          <span class="rank-no">{{ index + 1 }}</span>
          <div>
            <strong>{{ translateModelName(row.model_name) }}</strong>
            <p>{{ translateGroup(row.group) }} | {{ translateBackbone(row.image_backbone) }} | {{ translateEncoder(row.sequence_encoder_type) }}</p>
          </div>
          <div class="rank-metric">
            <span>{{ metricLabel(sortMetric) }}</span>
            <strong>{{ formatMetric(row[sortMetric]) }}</strong>
          </div>
        </div>
      </div>
      <p v-else class="empty">暂无符合条件的实验结果。</p>
    </article>

    <article class="panel insight-panel">
      <div class="panel-head">
        <h2>实验解读</h2>
      </div>
      <div v-if="highlightRows.length" class="headline-list">
        <div v-for="(line, index) in highlightRows" :key="`${index}-${line}`" class="headline-card">
          <strong>核心结论 {{ index + 1 }}</strong>
          <p>{{ line }}</p>
        </div>
      </div>
      <div v-if="filteredRows.length" class="insight-list">
        <div v-for="row in filteredRows" :key="row.model_name" class="insight-card">
          <strong>{{ translateModelName(row.model_name) }}</strong>
          <p>骨干网络: {{ translateBackbone(row.image_backbone) }}，序列编码: {{ translateEncoder(row.sequence_encoder_type) }}</p>
          <p>命中率 Hit@10: {{ formatMetric(row['hit@10']) }}</p>
          <p>NDCG@10: {{ formatMetric(row['ndcg@10']) }}</p>
        </div>
      </div>
      <p v-else class="empty">暂无实验汇总结果。</p>
    </article>

    <article class="panel chart-panel">
      <div class="panel-head">
        <h2>实验图表</h2>
      </div>
      <div v-if="charts.length" class="chart-grid">
        <figure v-for="chart in charts" :key="chart.name" class="chart-card">
          <img :src="chart.url" :alt="translateMetric(chart.metric)" />
          <figcaption>
            <strong>{{ translateMetric(chart.metric) }}</strong>
            <span>{{ translateChartCaption(chart.caption) }}</span>
          </figcaption>
        </figure>
      </div>
      <p v-else class="empty">暂无实验图表，请先运行对比汇总脚本。</p>
    </article>
  </section>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import ExperimentComparisonPanel from '../components/ExperimentComparisonPanel.vue'
import { fetchJson, formatMetric } from '../lib/api'

const experiments = ref(null)
const selectedGroup = ref('all')
const selectedBackbone = ref('all')
const selectedEncoder = ref('all')
const sortMetric = ref('hit@10')

const rows = computed(() => experiments.value?.summary?.results || [])
const highlightRows = computed(() => experiments.value?.charts?.highlights || experiments.value?.insights?.highlights || [])
const summaryCards = computed(() => experiments.value?.charts?.summary_cards || experiments.value?.insights?.summary_cards || [])
const charts = computed(() => experiments.value?.charts?.charts || [])

const availableGroups = computed(() => [...new Set(rows.value.map((row) => row.group).filter(Boolean))])
const availableBackbones = computed(() => [...new Set(rows.value.map((row) => row.image_backbone).filter(Boolean))])
const availableEncoders = computed(() => [...new Set(rows.value.map((row) => row.sequence_encoder_type).filter(Boolean))])

const filteredRows = computed(() => {
  const output = rows.value.filter((row) => {
    const groupOk = selectedGroup.value === 'all' || row.group === selectedGroup.value
    const backboneOk = selectedBackbone.value === 'all' || row.image_backbone === selectedBackbone.value
    const encoderOk = selectedEncoder.value === 'all' || row.sequence_encoder_type === selectedEncoder.value
    return groupOk && backboneOk && encoderOk
  })

  return [...output].sort((a, b) => Number(b[sortMetric.value] ?? -1) - Number(a[sortMetric.value] ?? -1))
})

const topRows = computed(() => filteredRows.value.slice(0, 5))
const filteredExperimentsPayload = computed(() => ({
  ...experiments.value,
  summary: {
    ...(experiments.value?.summary || {}),
    results: filteredRows.value,
  },
}))

async function loadData() {
  experiments.value = await fetchJson('/api/experiments/')
}

function translateGroup(value) {
  const text = String(value || '-')
  return text
    .replace('baseline', '基线模型')
    .replace('sequence', '序列模型')
    .replace('ablation', '消融实验')
}

function translateBackbone(value) {
  const text = String(value || '-')
  if (text === '-' || text === 'all') return text === 'all' ? '全部' : '-'
  return text
    .replace('resnet50', 'ResNet50')
    .replace('vit_b_16', 'ViT-B/16')
}

function translateEncoder(value) {
  const text = String(value || '-')
  if (text === '-' || text === 'all') return text === 'all' ? '全部' : '-'
  return text
    .replace('transformer', 'Transformer')
    .replace('gru', 'GRU')
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

function translateSummaryTitle(value) {
  const text = String(value || '')
  return text
    .replace('Best Model', '最佳模型')
    .replace('Best Overall', '最佳模型')
    .replace('Best Baseline', '最佳基线')
    .replace('Main Model Gain', '主模型提升')
    .replace('Best Backbone', '最佳视觉骨干')
}

function translateCardValue(title, value) {
  const text = String(value || '')
  if (String(title || '').includes('Backbone')) {
    return translateBackbone(text)
  }
  return translateModelName(text)
}

function translateSummaryDetail(value) {
  return String(value || '')
    .replace('Hit@10 gain', '命中率 Hit@10 提升')
    .replace('Relative gain over Gated Fusion on Hit@10 / NDCG@10', '相对 Gated Fusion 的 Hit@10 / NDCG@10 提升')
}

function translateMetric(value) {
  const text = String(value || '')
  return text
    .replace('hit@10', '命中率 Hit@10')
    .replace('ndcg@10', 'NDCG@10')
    .replace('ablation', '消融实验对比')
}

function metricLabel(value) {
  return translateMetric(value)
}

function translateChartCaption(value) {
  const text = String(value || '').trim()
  return text || '实验结果图表，可用于展示模型间性能差异。'
}

onMounted(loadData)
</script>

<style scoped>
.experiment-layout {
  display: grid;
  grid-template-columns: 1.1fr 0.9fr;
  gap: 20px;
}
.panel {
  background: rgba(255,255,255,0.88);
  border: 1px solid var(--line);
  border-radius: 24px;
  padding: 20px;
  box-shadow: 0 18px 40px rgba(17,24,39,0.06);
}
.briefing-panel,
.controls-panel,
.card-panel,
.chart-panel {
  grid-column: 1 / -1;
}
.panel-head {
  margin-bottom: 16px;
}
.panel-head h2 {
  margin: 0;
  font-size: 22px;
}
.briefing-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
}
.brief-card {
  padding: 18px;
  border-radius: 18px;
  border: 1px solid rgba(180,83,9,0.12);
  background: linear-gradient(135deg, #fff7ed, #ffffff);
}
.brief-card span {
  display: inline-block;
  margin-bottom: 8px;
  color: var(--muted);
  font-size: 12px;
}
.brief-card strong {
  display: block;
  margin-bottom: 8px;
  font-size: 18px;
  line-height: 1.4;
}
.brief-card p {
  margin: 0;
  color: var(--muted);
  line-height: 1.7;
}
.control-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
}
label span {
  display: block;
  margin-bottom: 8px;
  font-size: 13px;
  color: var(--muted);
}
select {
  width: 100%;
  padding: 12px 14px;
  border-radius: 14px;
  border: 1px solid var(--line);
  background: #fff;
}
.hint {
  margin: 14px 0 0;
  color: var(--muted);
}
.summary-grid {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 14px;
}
.summary-card {
  padding: 18px;
  border-radius: 18px;
  border: 1px solid rgba(15,118,110,0.12);
  background: linear-gradient(135deg, #ecfeff, #f8fafc);
}
.summary-card span {
  display: inline-block;
  margin-bottom: 10px;
  color: var(--muted);
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.summary-card strong {
  display: block;
  margin-bottom: 8px;
  font-size: 20px;
  line-height: 1.3;
}
.summary-card p {
  margin: 0;
  color: var(--muted);
  line-height: 1.65;
}
.ranking-list,
.insight-list,
.headline-list {
  display: grid;
  gap: 12px;
}
.ranking-card,
.insight-card,
.headline-card {
  display: grid;
  gap: 10px;
  padding: 16px;
  border-radius: 18px;
  background: #fff;
  border: 1px solid var(--line);
}
.ranking-card {
  grid-template-columns: auto 1fr auto;
  align-items: center;
}
.rank-no {
  width: 38px;
  height: 38px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  background: linear-gradient(135deg, var(--accent), #115e59);
  color: #fff;
  font-weight: 700;
}
.ranking-card p,
.insight-card p,
.headline-card p {
  margin: 0;
  color: var(--muted);
  line-height: 1.65;
}
.rank-metric {
  text-align: right;
}
.rank-metric span {
  display: block;
  color: var(--muted);
  font-size: 12px;
}
.chart-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
  gap: 16px;
}
.chart-card {
  margin: 0;
  padding: 14px;
  border-radius: 18px;
  border: 1px solid var(--line);
  background: #fff;
}
.chart-card img {
  width: 100%;
  display: block;
  border-radius: 12px;
  background: #f8fafc;
}
.chart-card figcaption {
  margin-top: 10px;
  display: grid;
  gap: 6px;
}
.chart-card figcaption span {
  color: var(--muted);
  line-height: 1.7;
}
.empty {
  color: var(--muted);
}
@media (max-width: 1100px) {
  .experiment-layout,
  .briefing-grid,
  .control-grid,
  .summary-grid {
    grid-template-columns: 1fr;
  }
  .ranking-card {
    grid-template-columns: 1fr;
  }
  .rank-metric {
    text-align: left;
  }
}
</style>
