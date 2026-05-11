<template>
  <article class="panel recommend-panel">
    <div class="panel-head">
      <h2>个性化推荐</h2>
      <span class="panel-tag">登录后可直接生成推荐</span>
    </div>

    <div class="auth-card">
      <template v-if="authState?.is_authenticated">
        <div class="auth-summary">
          <div>
            <p class="auth-label">当前账号</p>
            <strong>{{ authState.display_name || authState.username }}</strong>
          </div>
          <span class="status-pill">已登录</span>
        </div>
        <p>系统会在后台自动关联推荐画像，不再展示底层推荐用户标识。</p>
        <div class="auth-actions">
          <button class="secondary" @click="$emit('use-bound-user')">生成我的推荐</button>
          <button class="ghost-btn" @click="toggleAdvanced">{{ showAdvanced ? '收起高级输入' : '高级输入' }}</button>
          <button class="ghost-btn" @click="$emit('logout')">退出登录</button>
        </div>
      </template>
      <template v-else>
        <div class="auth-tabs">
          <button :class="{ active: authMode === 'login' }" @click="$emit('update:authMode', 'login')">登录</button>
          <button :class="{ active: authMode === 'register' }" @click="$emit('update:authMode', 'register')">注册</button>
        </div>
        <div class="auth-form">
          <input :value="authForm.username" placeholder="输入账号名" @input="$emit('update:authForm', { ...authForm, username: $event.target.value })" />
          <input :value="authForm.password" type="password" placeholder="输入密码" @input="$emit('update:authForm', { ...authForm, password: $event.target.value })" />
          <input v-if="authMode === 'register'" :value="authForm.display_name" placeholder="昵称，可选" @input="$emit('update:authForm', { ...authForm, display_name: $event.target.value })" />
          <button @click="$emit(authMode === 'login' ? 'login' : 'register')">{{ authMode === 'login' ? '登录并进入推荐' : '注册并进入推荐' }}</button>
        </div>
      </template>
    </div>

    <div v-if="!authState?.is_authenticated || showAdvanced" class="controls">
      <input
        :value="userId"
        :placeholder="authState?.is_authenticated ? '高级模式：手动输入 user_id 进行调试' : '输入 user_id，例如 A1KLRMWW2FWPL4'"
        @input="$emit('update:userId', $event.target.value)"
        @keyup.enter="$emit('recommend')"
      />
      <button @click="$emit('recommend')">生成推荐</button>
    </div>
    <div v-else class="quick-actions">
      <button class="primary-btn" @click="$emit('use-bound-user')">一键生成推荐</button>
      <p>已为当前账号准备专属推荐画像，你可以直接获取 Top 10 商品建议。</p>
    </div>

    <p v-if="errorMessage" class="error">{{ errorMessage }}</p>
    <div v-if="userProfile" class="profile-card">
      <div class="profile-title">
        <strong>推荐画像已加载</strong>
        <span>基于历史行为自动建模</span>
      </div>
      <div class="profile-metrics">
        <div>
          <span>历史行为数</span>
          <strong>{{ userProfile.behavior_summary?.behavior_count ?? 0 }}</strong>
        </div>
        <div>
          <span>平均评分</span>
          <strong>{{ formatMetric(userProfile.behavior_summary?.mean_score) }}</strong>
        </div>
        <div>
          <span>最近交互</span>
          <strong>{{ formatDate(userProfile.behavior_summary?.last_behavior_at) }}</strong>
        </div>
      </div>
    </div>

    <div class="cards" v-if="items.length">
      <button v-for="item in items" :key="item.item_id" class="card action-card" @click="$emit('select-product', item.item_id)">
        <img v-if="item.has_image !== false" :src="item.image_url || placeholderImage" @error="onImageError($event)" />
        <div v-else class="placeholder-cover">
          <span class="placeholder-badge">暂无图片</span>
          <strong>{{ shortTitle(item.title || item.item_id) }}</strong>
          <small>推荐结果</small>
        </div>
        <div class="card-body">
          <div class="card-headline">
            <h3>{{ item.title || item.item_id }}</h3>
            <span class="score-chip">{{ Number(item.score || 0).toFixed(3) }}</span>
          </div>
          <p>商品ID: {{ item.item_id }}</p>
          <p>评分: {{ item.rating ?? '-' }}</p>
        </div>
      </button>
    </div>
    <div v-else class="empty-state">
      <strong>还没有推荐结果</strong>
      <p>登录账号后可直接生成推荐，也可以手动输入 `user_id` 进行调试。</p>
    </div>
  </article>
</template>

<script setup>
import { ref } from 'vue'
import { formatDate, formatMetric, placeholderImage } from '../lib/api'

const showAdvanced = ref(false)

defineProps({
  userId: { type: String, default: '' },
  items: { type: Array, default: () => [] },
  errorMessage: { type: String, default: '' },
  userProfile: { type: Object, default: null },
  authState: { type: Object, default: null },
  authMode: { type: String, default: 'login' },
  authForm: { type: Object, default: () => ({}) },
})

defineEmits(['update:userId', 'recommend', 'select-product', 'login', 'register', 'logout', 'use-bound-user', 'update:authMode', 'update:authForm'])

function onImageError(event) {
  event.target.onerror = null
  event.target.alt = ''
  event.target.src = placeholderImage
}

function shortTitle(value) {
  const text = String(value || '').trim()
  return text ? text.slice(0, 18) : '商品'
}

function toggleAdvanced() {
  showAdvanced.value = !showAdvanced.value
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
  gap: 12px;
  margin-bottom: 16px;
}
.panel-head h2 { margin: 0; font-size: 22px; }
.panel-tag {
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 12px;
  color: var(--accent);
  background: rgba(15,118,110,0.12);
}
.auth-card {
  margin-bottom: 16px;
  padding: 16px;
  border-radius: 18px;
  background: rgba(15,118,110,0.06);
}
.auth-summary {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
}
.auth-label {
  margin: 0 0 6px;
  color: var(--muted);
  font-size: 12px;
}
.status-pill {
  padding: 6px 10px;
  border-radius: 999px;
  background: #ecfdf5;
  color: #047857;
  font-size: 12px;
}
.auth-card p { margin: 8px 0 0; color: var(--muted); }
.auth-tabs { display: flex; gap: 8px; margin-bottom: 12px; }
.auth-tabs button, .auth-actions button {
  border: 1px solid var(--line);
  border-radius: 12px;
  padding: 10px 12px;
  cursor: pointer;
  background: #fff;
  color: var(--ink);
}
.auth-tabs button.active {
  background: linear-gradient(135deg, var(--accent), #115e59);
  color: #fff;
  border-color: transparent;
}
.auth-form {
  display: grid;
  gap: 10px;
}
.auth-form input, .controls input {
  flex: 1;
  padding: 14px;
  border-radius: 14px;
  border: 1px solid var(--line);
}
.auth-form button, .controls button, .secondary, .primary-btn {
  border: 0;
  border-radius: 14px;
  padding: 14px 18px;
  background: linear-gradient(135deg, var(--accent), #115e59);
  color: #fff;
  cursor: pointer;
}
.ghost-btn {
  background: transparent !important;
  color: var(--ink) !important;
}
.auth-actions {
  display: flex;
  gap: 10px;
  margin-top: 14px;
  flex-wrap: wrap;
}
.controls { display: flex; gap: 12px; margin-bottom: 16px; }
.quick-actions {
  margin-bottom: 16px;
  padding: 16px;
  border-radius: 18px;
  border: 1px dashed rgba(15,118,110,0.3);
  background: rgba(255,255,255,0.75);
}
.quick-actions p {
  margin: 10px 0 0;
  color: var(--muted);
}
.error { color: #b42318; }
.profile-card {
  margin-bottom: 16px;
  padding: 16px;
  border-radius: 18px;
  background: rgba(15,118,110,0.06);
}
.profile-title {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  margin-bottom: 14px;
}
.profile-title span {
  color: var(--muted);
  font-size: 12px;
}
.profile-metrics {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
}
.profile-metrics div {
  padding: 12px;
  border-radius: 16px;
  background: rgba(255,255,255,0.8);
  border: 1px solid rgba(15,118,110,0.08);
}
.profile-metrics span {
  display: block;
  color: var(--muted);
  font-size: 12px;
}
.profile-metrics strong {
  display: block;
  margin-top: 8px;
  font-size: 16px;
}
.cards {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 16px;
}
.card {
  overflow: hidden;
  border-radius: 18px;
  background: #fff;
  border: 1px solid var(--line);
}
.action-card {
  width: 100%;
  display: grid;
  grid-template-columns: 132px 1fr;
  align-items: stretch;
  padding: 0;
  cursor: pointer;
  text-align: left;
  transition: transform 0.18s ease, box-shadow 0.18s ease;
}
.action-card.featured {
  grid-template-columns: 148px 1fr;
  border-color: rgba(15,118,110,0.22);
  box-shadow: 0 20px 34px rgba(15,23,42,0.08);
}
.action-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 18px 30px rgba(15,23,42,0.08);
}
.card-media {
  position: relative;
  width: 132px;
  min-height: 164px;
  border-right: 1px solid var(--line);
}
.featured .card-media {
  width: 148px;
}
.featured-badge {
  position: absolute;
  left: 10px;
  top: 10px;
  z-index: 1;
  padding: 5px 10px;
  border-radius: 999px;
  background: rgba(15,118,110,0.9);
  color: #fff;
  font-size: 12px;
}
.card img,
.placeholder-cover {
  width: 100%;
  height: 100%;
  min-height: 164px;
  background: linear-gradient(145deg, #eef2f7, #dfe7ef);
}
.card img {
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
.card-body {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  gap: 12px;
  padding: 16px;
}
.meta-chips {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
}
.meta-chip {
  padding: 5px 10px;
  border-radius: 999px;
  background: rgba(15,118,110,0.08);
  color: var(--accent);
  font-size: 12px;
}
.card-headline {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  align-items: start;
}
.card-body h3 {
  margin: 0 0 10px;
  font-size: 16px;
  line-height: 1.45;
  flex: 1;
}
.score-chip {
  padding: 4px 8px;
  border-radius: 999px;
  background: rgba(15,118,110,0.1);
  color: var(--accent);
  font-size: 12px;
}
.card-body p { margin: 4px 0; font-size: 13px; color: var(--muted); }
.empty-state {
  padding: 20px;
  border-radius: 18px;
  border: 1px dashed var(--line);
  color: var(--muted);
  background: rgba(255,255,255,0.7);
}
.empty-state strong {
  display: block;
  margin-bottom: 8px;
  color: var(--ink);
}
@media (max-width: 960px) {
  .profile-metrics { grid-template-columns: 1fr; }
  .cards {
    grid-template-columns: 1fr;
  }
  .action-card,
  .action-card.featured {
    grid-template-columns: 128px 1fr;
  }
  .card-media,
  .featured .card-media {
    width: 128px;
  }
}
@media (max-width: 640px) {
  .controls, .auth-actions, .panel-head, .auth-summary, .profile-title { flex-direction: column; align-items: stretch; }
  .action-card,
  .action-card.featured {
    grid-template-columns: 116px 1fr;
  }
  .card-media,
  .featured .card-media {
    width: 116px;
  }
}
</style>
