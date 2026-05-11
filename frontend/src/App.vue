<template>
  <div class="page-shell">
    <header class="hero">
      <div>
        <p class="eyebrow">Vue Frontend + Django Backend</p>
        <h1>多模态商品推荐系统</h1>
        <p class="intro">系统集成商品文本、图像与用户行为序列建模能力，支持实验分析、用户画像、推荐服务与架构说明展示。</p>
        <div v-if="heroCards.length" class="hero-highlights">
          <article v-for="card in heroCards" :key="card.title" class="hero-card">
            <span>{{ translateSummaryTitle(card.title) }}</span>
            <strong>{{ card.value }}</strong>
            <p>{{ translateSummaryDetail(card.detail) }}</p>
          </article>
        </div>
      </div>
      <nav class="nav-tabs">
        <RouterLink to="/" class="nav-link">系统总览</RouterLink>
        <RouterLink to="/experiments" class="nav-link">实验分析</RouterLink>
        <RouterLink to="/users" class="nav-link">用户画像</RouterLink>
        <RouterLink to="/architecture" class="nav-link">架构说明</RouterLink>
      </nav>
    </header>

    <section class="overview-strip">
      <article>
        <span>研究目标</span>
        <strong>多模态表示学习</strong>
        <p>融合商品文本、图像与用户行为序列，构建可用于推荐的统一表示空间。</p>
      </article>
      <article>
        <span>实验能力</span>
        <strong>多模型对比分析</strong>
        <p>支持仅文本、仅图像、早期融合与序列注意力模型等多种实验对比。</p>
      </article>
      <article>
        <span>系统实现</span>
        <strong>Django + Vue 前后端</strong>
        <p>覆盖商品检索、推荐展示、用户画像、实验报告与系统部署说明。</p>
      </article>
    </section>

    <RouterView />
  </div>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { RouterLink, RouterView } from 'vue-router'
import { fetchJson } from './lib/api'

const experimentPayload = ref(null)
const summaryCards = computed(() => experimentPayload.value?.charts?.summary_cards || experimentPayload.value?.insights?.summary_cards || [])
const heroCards = computed(() => summaryCards.value.slice(0, 3))

async function loadHeroCards() {
  try {
    experimentPayload.value = await fetchJson('/api/experiments/')
  } catch {
    experimentPayload.value = null
  }
}

function translateSummaryTitle(value) {
  const text = String(value || '')
  return text
    .replace('Best Model', '最佳模型')
    .replace('Best Baseline', '最佳基线')
    .replace('Main Model Gain', '主模型提升')
    .replace('Best Backbone', '最佳视觉骨干')
}

function translateSummaryDetail(value) {
  const text = String(value || '')
  return text
    .replace('Main model outperforms baseline results in the current summary.', '当前主模型相较基线模型取得了更优效果。')
    .replace('Best-performing experiment in the current report.', '当前实验报告中的综合最优结果。')
    .replace('Strongest baseline model under the current configuration.', '当前配置下表现最好的基线模型。')
    .replace('Visual backbone with the strongest overall performance.', '当前实验中综合表现最好的视觉骨干网络。')
}

onMounted(loadHeroCards)
</script>

<style>
:root {
  color-scheme: light;
  --bg: #f2efe8;
  --panel: rgba(255,255,255,0.88);
  --ink: #1f2937;
  --muted: #667085;
  --accent: #0f766e;
  --accent-soft: rgba(15,118,110,0.10);
  --line: #d7dde5;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Segoe UI", "Microsoft YaHei", sans-serif;
  color: var(--ink);
  background:
    radial-gradient(circle at top left, rgba(180,83,9,0.12), transparent 28%),
    radial-gradient(circle at bottom right, rgba(15,118,110,0.16), transparent 30%),
    var(--bg);
}
.page-shell {
  max-width: 1240px;
  margin: 0 auto;
  padding: 32px 20px 48px;
}
.hero {
  display: flex;
  justify-content: space-between;
  gap: 24px;
  align-items: end;
  margin-bottom: 24px;
}
.eyebrow {
  display: inline-block;
  margin: 0 0 12px;
  padding: 6px 12px;
  border-radius: 999px;
  background: var(--accent-soft);
  color: var(--accent);
  font-size: 12px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
h1 {
  margin: 0;
  font-size: 42px;
  line-height: 1.08;
}
.intro {
  max-width: 760px;
  margin: 12px 0 0;
  color: var(--muted);
  line-height: 1.8;
}
.hero-highlights {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
  margin-top: 18px;
}
.hero-card {
  padding: 14px 16px;
  border-radius: 18px;
  background: rgba(255,255,255,0.82);
  border: 1px solid rgba(15,118,110,0.12);
  box-shadow: 0 12px 24px rgba(15,23,42,0.04);
}
.hero-card span {
  display: inline-block;
  margin-bottom: 8px;
  color: var(--muted);
  font-size: 12px;
}
.hero-card strong {
  display: block;
  font-size: 18px;
  margin-bottom: 6px;
}
.hero-card p {
  margin: 0;
  color: var(--muted);
  line-height: 1.6;
  font-size: 13px;
}
.nav-tabs {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  justify-content: flex-end;
}
.nav-link {
  padding: 10px 14px;
  border-radius: 999px;
  background: rgba(255,255,255,0.7);
  border: 1px solid var(--line);
  font-size: 13px;
  color: var(--ink);
  text-decoration: none;
}
.nav-link.router-link-active {
  background: linear-gradient(135deg, var(--accent), #115e59);
  color: #fff;
  border-color: transparent;
}
.overview-strip {
  display: grid;
  grid-template-columns: repeat(3, 1fr);
  gap: 16px;
  margin-bottom: 24px;
}
.overview-strip article {
  padding: 18px;
  border-radius: 22px;
  background: rgba(255,255,255,0.82);
  border: 1px solid var(--line);
  box-shadow: 0 18px 40px rgba(17,24,39,0.05);
}
.overview-strip span {
  display: inline-block;
  font-size: 12px;
  color: var(--muted);
  margin-bottom: 10px;
}
.overview-strip strong {
  display: block;
  font-size: 22px;
  margin-bottom: 8px;
}
.overview-strip p {
  margin: 0;
  color: var(--muted);
  line-height: 1.7;
}
@media (max-width: 900px) {
  .hero { flex-direction: column; align-items: start; }
  h1 { font-size: 32px; }
  .overview-strip, .hero-highlights { grid-template-columns: 1fr; }
}
</style>

