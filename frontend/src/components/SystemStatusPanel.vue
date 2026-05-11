<template>
  <article class="panel status-panel">
    <div class="panel-head">
      <h2>&#31995;&#32479;&#29366;&#24577;</h2>
      <button class="ghost" @click="$emit('refresh')">&#21047;&#26032;</button>
    </div>
    <div v-if="status" class="status-grid">
      <div class="metric"><span>&#29992;&#25143;&#25968;</span><strong>{{ status.users }}</strong></div>
      <div class="metric"><span>&#21830;&#21697;&#25968;</span><strong>{{ status.products }}</strong></div>
      <div class="metric"><span>&#34892;&#20026;&#25968;</span><strong>{{ status.behaviors }}</strong></div>
      <div class="metric"><span>&#26085;&#24535;&#25968;</span><strong>{{ status.recommendation_logs }}</strong></div>
    </div>
    <div v-if="status" class="meta-list">
      <p>&#25968;&#25454;&#24211;: {{ status.database_engine }}</p>
      <p>&#32531;&#23384;: {{ status.cache_backend }}</p>
      <p>&#27169;&#22411;&#25991;&#20214;: {{ status.artifacts.model_exists ? '&#24050;&#23601;&#32490;' : '&#32570;&#22833;' }}</p>
    </div>
  </article>
</template>

<script setup>
defineProps({
  status: { type: Object, default: null },
})

defineEmits(['refresh'])
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
.status-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 12px;
}
.metric {
  padding: 14px;
  border-radius: 18px;
  background: #fff;
  border: 1px solid var(--line);
}
.metric span { display: block; font-size: 12px; color: var(--muted); }
.metric strong { display: block; margin-top: 8px; font-size: 24px; }
.meta-list { margin-top: 16px; color: var(--muted); line-height: 1.8; }
@media (max-width: 640px) {
  .status-grid { grid-template-columns: 1fr; }
}
</style>
