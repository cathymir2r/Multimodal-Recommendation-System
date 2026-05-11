<template>
  <section class="architecture-layout">
    <article class="panel full-width intro-panel">
      <div class="panel-head">
        <h2>系统架构说明</h2>
      </div>
      <p class="intro-text">本系统围绕多模态商品表示学习、个性化推荐建模与前后端平台实现三部分展开，兼顾算法研究、系统实现与展示应用三个层面的需求。</p>
      <div class="intro-grid">
        <div class="intro-card">
          <span>数据基础</span>
          <strong>多模态特征构建</strong>
          <p>商品标题、描述、评论文本与图像信息共同参与表示学习，为推荐模型提供统一的输入特征。</p>
        </div>
        <div class="intro-card">
          <span>模型方法</span>
          <strong>个性化推荐建模</strong>
          <p>用户历史行为序列与商品多模态表示共同进入推荐模型，用于刻画兴趣变化与偏好差异。</p>
        </div>
        <div class="intro-card">
          <span>系统实现</span>
          <strong>前后端协同展示</strong>
          <p>Django 负责后端接口与数据服务，Vue 负责页面展示与交互，实现推荐、分析与架构展示的一体化运行。</p>
        </div>
      </div>
    </article>

    <article class="panel">
      <div class="panel-head">
        <h2>研究流程</h2>
      </div>
      <div class="timeline">
        <div v-for="(step, index) in architecture?.research_pipeline || []" :key="step.stage" class="timeline-item">
          <span class="timeline-no">0{{ index + 1 }}</span>
          <div>
            <strong>{{ step.stage }}</strong>
            <p>{{ step.detail }}</p>
          </div>
        </div>
      </div>
    </article>

    <article class="panel">
      <div class="panel-head">
        <h2>系统模块</h2>
      </div>
      <div class="module-grid">
        <div v-for="module in architecture?.system_modules || []" :key="module.name" class="module-card">
          <strong>{{ module.name }}</strong>
          <p>{{ module.detail }}</p>
        </div>
      </div>
    </article>

    <article class="panel full-width">
      <div class="panel-head">
        <h2>部署栈</h2>
      </div>
      <div class="deploy-grid">
        <div v-for="item in architecture?.deployment || []" :key="item.layer" class="deploy-card">
          <span>{{ item.layer }}</span>
          <strong>{{ item.tech }}</strong>
        </div>
      </div>
    </article>
  </section>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { fetchJson } from '../lib/api'

const architecture = ref(null)

async function loadArchitecture() {
  architecture.value = await fetchJson('/api/architecture/')
}

onMounted(loadArchitecture)
</script>

<style scoped>
.architecture-layout {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 20px;
}
.panel {
  background: rgba(255,255,255,0.88);
  border: 1px solid var(--line);
  border-radius: 24px;
  padding: 20px;
  box-shadow: 0 18px 40px rgba(17,24,39,0.06);
}
.full-width { grid-column: 1 / -1; }
.panel-head { margin-bottom: 16px; }
.panel-head h2 { margin: 0; font-size: 22px; }
.intro-text {
  margin: 0 0 18px;
  color: var(--muted);
  line-height: 1.9;
}
.intro-grid {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 14px;
}
.intro-card {
  padding: 16px;
  border-radius: 18px;
  background: #fff;
  border: 1px solid var(--line);
}
.intro-card span {
  display: inline-block;
  margin-bottom: 8px;
  color: var(--muted);
  font-size: 12px;
}
.intro-card strong {
  display: block;
  margin-bottom: 8px;
  font-size: 18px;
  line-height: 1.4;
}
.intro-card p {
  margin: 0;
  color: var(--muted);
  line-height: 1.7;
}
.timeline { display: grid; gap: 14px; }
.timeline-item {
  display: grid;
  grid-template-columns: auto 1fr;
  gap: 12px;
  align-items: start;
  padding: 16px;
  border-left: 4px solid var(--accent);
  border-radius: 14px;
  background: #fff;
  border: 1px solid var(--line);
}
.timeline-no {
  width: 36px;
  height: 36px;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  border-radius: 999px;
  background: rgba(15,118,110,0.12);
  color: var(--accent);
  font-weight: 700;
}
.timeline-item strong { display: block; margin-bottom: 8px; }
.timeline-item p { margin: 0; color: var(--muted); line-height: 1.7; }
.module-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 14px;
}
.module-card,
.deploy-card {
  padding: 16px;
  border-radius: 18px;
  background: #fff;
  border: 1px solid var(--line);
}
.module-card strong,
.deploy-card strong { display: block; margin-bottom: 8px; }
.module-card p { margin: 0; color: var(--muted); line-height: 1.7; }
.deploy-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 14px;
}
.deploy-card span {
  display: block;
  color: var(--muted);
  margin-bottom: 8px;
  font-size: 12px;
}
@media (max-width: 900px) {
  .architecture-layout,
  .intro-grid,
  .module-grid {
    grid-template-columns: 1fr;
  }
  .timeline-item {
    grid-template-columns: 1fr;
  }
}
</style>
