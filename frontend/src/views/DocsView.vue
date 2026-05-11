<template>
  <section class="docs-layout">
    <article class="panel docs-panel">
      <div class="panel-head">
        <h2>API Docs</h2>
      </div>
      <div v-for="group in docs?.groups || []" :key="group.name" class="group-block">
        <h3>{{ group.name }}</h3>
        <div v-for="endpoint in group.endpoints" :key="`${endpoint.method}-${endpoint.path}`" class="endpoint-card">
          <strong>{{ endpoint.method }} {{ endpoint.path }}</strong>
          <p>{{ endpoint.description }}</p>
        </div>
      </div>
    </article>

    <article class="panel export-panel">
      <div class="panel-head actions-head">
        <h2>文档导出</h2>
        <div class="action-row">
          <button class="switch-btn" :class="{ active: exportMode === 'summary' }" @click="exportMode = 'summary'">摘要</button>
          <button class="switch-btn" :class="{ active: exportMode === 'appendix' }" @click="exportMode = 'appendix'">附录</button>
          <button class="download-btn" @click="downloadCurrent">下载</button>
        </div>
      </div>
      <p class="export-tip">{{ exportMode === 'summary' ? '适合答辩展示和项目简介。' : '适合论文附录、方法说明和实验配置记录。' }}</p>
      <pre class="export-box">{{ exportMode === 'summary' ? summaryText : appendixText }}</pre>
    </article>
  </section>
</template>

<script setup>
import { computed, onMounted, ref } from 'vue'
import { fetchJson } from '../lib/api'

const docs = ref(null)
const summaryText = ref('')
const appendixText = ref('')
const exportMode = ref('summary')

const currentFilename = computed(() => {
  return exportMode.value === 'summary' ? 'multimodal_system_summary.md' : 'multimodal_system_appendix.md'
})

async function loadData() {
  docs.value = await fetchJson('/api/docs/')
  const [summaryResponse, appendixResponse] = await Promise.all([
    fetch('/api/export/summary/'),
    fetch('/api/export/appendix/'),
  ])
  summaryText.value = await summaryResponse.text()
  appendixText.value = await appendixResponse.text()
}

function downloadCurrent() {
  const content = exportMode.value === 'summary' ? summaryText.value : appendixText.value
  const blob = new Blob([content], { type: 'text/plain;charset=utf-8' })
  const url = URL.createObjectURL(blob)
  const link = document.createElement('a')
  link.href = url
  link.download = currentFilename.value
  link.click()
  URL.revokeObjectURL(url)
}

onMounted(loadData)
</script>

<style scoped>
.docs-layout {
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
.panel-head {
  margin-bottom: 16px;
}
.actions-head {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
}
.panel-head h2 {
  margin: 0;
  font-size: 22px;
}
.action-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
}
.switch-btn,
.download-btn {
  border: 0;
  border-radius: 14px;
  padding: 12px 16px;
  cursor: pointer;
}
.switch-btn {
  background: #e5e7eb;
  color: #1f2937;
}
.switch-btn.active {
  background: linear-gradient(135deg, #b45309, #92400e);
  color: #fff;
}
.download-btn {
  background: linear-gradient(135deg, var(--accent), #115e59);
  color: #fff;
}
.export-tip {
  margin: 0 0 14px;
  color: var(--muted);
}
.group-block + .group-block {
  margin-top: 18px;
}
.group-block h3 {
  margin: 0 0 10px;
}
.endpoint-card {
  padding: 14px;
  border: 1px solid var(--line);
  border-radius: 16px;
  background: #fff;
}
.endpoint-card + .endpoint-card {
  margin-top: 10px;
}
.endpoint-card p {
  margin: 8px 0 0;
  color: var(--muted);
}
.export-box {
  min-height: 560px;
  margin: 0;
  padding: 16px;
  border-radius: 16px;
  background: #111827;
  color: #f9fafb;
  overflow: auto;
  white-space: pre-wrap;
  line-height: 1.7;
}
@media (max-width: 980px) {
  .docs-layout {
    grid-template-columns: 1fr;
  }
  .actions-head {
    flex-direction: column;
    align-items: flex-start;
  }
}
</style>
