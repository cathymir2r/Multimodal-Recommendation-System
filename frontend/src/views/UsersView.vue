<template>
  <section class="page-stack">
    <article class="panel mapping-panel">
      <div class="panel-head">
        <h2>&#36134;&#21495;&#19982;&#30011;&#20687;&#23545;&#24212;</h2>
        <button class="ghost" @click="loadBindings">&#21047;&#26032;</button>
      </div>
      <div v-if="bindings.length" class="binding-grid">
        <button
          v-for="binding in bindings"
          :key="`${binding.username}-${binding.recommendation_user_id}`"
          class="binding-card"
          @click="selectBinding(binding.recommendation_user_id)"
        >
          <strong>{{ binding.display_name || binding.username }}</strong>
          <p>&#36134;&#21495;: {{ binding.username }}</p>
          <p>&#30011;&#20687; user_id: {{ binding.recommendation_user_id }}</p>
        </button>
      </div>
      <p v-else class="empty">&#26242;&#26080;&#36134;&#21495;&#26144;&#23556;&#20449;&#24687;&#12290;</p>
    </article>

    <section class="view-grid">
      <article class="panel search-panel">
        <div class="panel-head">
          <h2>&#29992;&#25143;&#30011;&#20687;&#26597;&#35810;</h2>
        </div>
        <div class="controls">
          <input v-model="userId" placeholder="&#36755;&#20837; user_id &#26597;&#35810;&#30011;&#20687;" @keyup.enter="loadUser" />
          <button @click="loadUser">&#26597;&#35810;</button>
        </div>
        <p v-if="errorMessage" class="error">{{ errorMessage }}</p>
        <div v-if="profile" class="profile-block">
          <h3>{{ profile.user_id }}</h3>
          <p>&#34892;&#20026;&#25968;: {{ profile.behavior_summary?.behavior_count ?? 0 }}</p>
          <p>&#24179;&#22343;&#35780;&#20998;: {{ formatMetric(profile.behavior_summary?.mean_score) }}</p>
          <p>&#26368;&#36817;&#34892;&#20026;: {{ formatDate(profile.behavior_summary?.last_behavior_at) }}</p>
        </div>
      </article>

      <article class="panel history-panel">
        <div class="panel-head">
          <h2>&#26368;&#36817;&#34892;&#20026;</h2>
        </div>
        <div v-if="profile?.recent_behaviors?.length" class="behavior-list">
          <div v-for="row in profile.recent_behaviors" :key="`${row.item_id}-${row.occurred_at}`" class="behavior-card">
            <strong>{{ row.title || row.item_id }}</strong>
            <p>&#31867;&#22411;: {{ row.behavior_type }}</p>
            <p>&#35780;&#20998;: {{ row.score ?? '-' }}</p>
            <p>&#26102;&#38388;: {{ formatDate(row.occurred_at) }}</p>
          </div>
        </div>
        <p v-else class="empty">&#26597;&#35810;&#29992;&#25143;&#21518;&#21487;&#26597;&#30475;&#26368;&#36817;&#34892;&#20026;&#12290;</p>
      </article>
    </section>
  </section>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import { fetchJson, formatDate, formatMetric } from '../lib/api'

const userId = ref('')
const profile = ref(null)
const errorMessage = ref('')
const bindings = ref([])

async function loadBindings() {
  const data = await fetchJson('/api/user-bindings/')
  bindings.value = data.items || []
}

async function loadUser() {
  errorMessage.value = ''
  profile.value = null
  if (!userId.value.trim()) {
    errorMessage.value = '请输入 user_id'
    return
  }
  try {
    profile.value = await fetchJson(`/api/users/${encodeURIComponent(userId.value.trim())}/`)
  } catch (error) {
    errorMessage.value = error.message
  }
}

function selectBinding(id) {
  userId.value = id
  loadUser()
}

onMounted(loadBindings)
</script>

<style scoped>
.page-stack {
  display: grid;
  gap: 20px;
}
.view-grid {
  display: grid;
  grid-template-columns: 0.9fr 1.1fr;
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
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
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
.binding-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 12px;
}
.binding-card {
  padding: 14px;
  border-radius: 16px;
  border: 1px solid var(--line);
  background: #fff;
  text-align: left;
  cursor: pointer;
  transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.binding-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 16px 28px rgba(15,23,42,0.08);
}
.binding-card strong {
  display: block;
  margin-bottom: 8px;
}
.binding-card p {
  margin: 4px 0;
  color: var(--muted);
}
.controls { display: flex; gap: 12px; margin-bottom: 16px; }
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
.error { color: #b42318; }
.profile-block {
  padding: 16px;
  border-radius: 18px;
  background: rgba(15,118,110,0.06);
}
.profile-block h3 { margin-top: 0; }
.profile-block p { color: var(--muted); }
.behavior-list { display: grid; gap: 12px; }
.behavior-card {
  padding: 14px;
  border-radius: 16px;
  background: #fff;
  border: 1px solid var(--line);
}
.behavior-card strong { display: block; margin-bottom: 8px; }
.behavior-card p { margin: 4px 0; color: var(--muted); }
.empty { color: var(--muted); }
@media (max-width: 900px) {
  .view-grid { grid-template-columns: 1fr; }
  .controls { flex-direction: column; }
}
</style>
