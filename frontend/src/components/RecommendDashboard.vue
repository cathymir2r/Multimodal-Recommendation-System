<template>
  <section class="layout">
    <SystemStatusPanel class="status-panel" :status="status" @refresh="loadAll" />
    <RecommendationWorkspace
      class="recommend-panel"
      :user-id="userId"
      :items="items"
      :error-message="errorMessage"
      :user-profile="userProfile"
      :auth-state="authState"
      :auth-mode="authMode"
      :auth-form="authForm"
      @update:user-id="userId = $event"
      @update:auth-mode="authMode = $event"
      @update:auth-form="authForm = $event"
      @recommend="recommend"
      @login="loginAccount"
      @register="registerAccount"
      @logout="logoutAccount"
      @use-bound-user="useBoundUser"
      @select-product="selectProduct"
    />
    <ProductDetailPanel class="detail-panel" :selected-product="selectedProduct" />
    <ExperimentComparisonPanel class="experiments-panel" :experiments="experiments" @refresh="loadExperiments" />
    <HistoryCatalogPanel
      class="history-catalog-panel"
      :history="history"
      :catalog-items="catalogItems"
      :catalog-query="catalogQuery"
      @refresh-history="loadHistory"
      @refresh-catalog="loadCatalog"
      @update:catalog-query="catalogQuery = $event"
      @select-product="selectProduct"
    />
  </section>
</template>

<script setup>
import { onMounted, ref } from 'vue'
import ExperimentComparisonPanel from './ExperimentComparisonPanel.vue'
import HistoryCatalogPanel from './HistoryCatalogPanel.vue'
import ProductDetailPanel from './ProductDetailPanel.vue'
import RecommendationWorkspace from './RecommendationWorkspace.vue'
import SystemStatusPanel from './SystemStatusPanel.vue'
import { fetchJson } from '../lib/api'

const userId = ref('')
const items = ref([])
const errorMessage = ref('')
const status = ref(null)
const experiments = ref(null)
const history = ref([])
const catalogItems = ref([])
const catalogQuery = ref('')
const selectedProduct = ref(null)
const userProfile = ref(null)
const authState = ref(null)
const authMode = ref('login')
const authForm = ref({ username: '', password: '', display_name: '' })
const recommendDebug = ref('idle')
const isRecommending = ref(false)

async function loadDashboard() {
  const data = await fetchJson('/api/dashboard/')
  status.value = data.status
  experiments.value = data.experiments
  history.value = data.history?.items || []
  catalogItems.value = data.catalog?.items || []
  authState.value = data.auth || authState.value
}

async function loadAuthState() {
  authState.value = await fetchJson('/api/auth/me/')
}

async function loadExperiments() {
  experiments.value = await fetchJson('/api/experiments/')
}

async function loadHistory() {
  const data = await fetchJson('/api/recommendations/history/?page_size=8')
  history.value = data.items || []
}

async function loadHistoryAndGetLatestResult(expectedUserId = '') {
  const data = await fetchJson('/api/recommendations/history/?page_size=8')
  history.value = data.items || []
  const latest = history.value.find(row => !expectedUserId || row.user_id === expectedUserId)
  return latest?.result_items || []
}

async function loadCatalog() {
  const query = catalogQuery.value.trim()
  const suffix = query ? `?q=${encodeURIComponent(query)}&page_size=8` : '?page_size=8'
  const data = await fetchJson(`/api/catalog/${suffix}`)
  catalogItems.value = data.items || []
}

async function loadUserProfile(id) {
  userProfile.value = await fetchJson(`/api/users/${encodeURIComponent(id)}/`)
}

async function selectProduct(itemId) {
  selectedProduct.value = await fetchJson(`/api/products/${encodeURIComponent(itemId)}/`)
}

function resetAuthForm() {
  authForm.value = { username: '', password: '', display_name: '' }
}

async function loginAccount() {
  errorMessage.value = ''
  try {
    authState.value = await fetchJson('/api/auth/login/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: authForm.value.username, password: authForm.value.password }),
    })
    userId.value = authState.value?.recommendation_user_id || ''
    resetAuthForm()
    if (userId.value) {
      await loadUserProfile(userId.value)
      await recommend()
    }
  } catch (error) {
    const message = error?.message || 'unknown'
    recommendDebug.value = `error:${message}`
    errorMessage.value = `login_error msg=${message}; debug=${recommendDebug.value}; bound=${authState.value?.recommendation_user_id || 'none'}; userId=${userId.value || 'empty'}`
  }
}

async function registerAccount() {
  errorMessage.value = ''
  try {
    authState.value = await fetchJson('/api/auth/register/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(authForm.value),
    })
    userId.value = authState.value?.recommendation_user_id || ''
    resetAuthForm()
    if (userId.value) {
      await loadUserProfile(userId.value)
      await recommend()
    }
  } catch (error) {
    const message = error?.message || 'unknown'
    recommendDebug.value = `error:${message}`
    errorMessage.value = `recommend_error msg=${message}; debug=${recommendDebug.value}; bound=${authState.value?.recommendation_user_id || 'none'}; userId=${userId.value || 'empty'}; items=${items.value.length}`
  }
}

async function logoutAccount() {
  await fetchJson('/api/auth/logout/', { method: 'POST' })
  authState.value = { is_authenticated: false }
  userId.value = ''
  userProfile.value = null
  items.value = []
}

async function useBoundUser() {
  userId.value = authState.value?.recommendation_user_id || ''
  await recommend()
}

async function recommend() {
  if (isRecommending.value) {
    recommendDebug.value = 'busy'
    return
  }

  errorMessage.value = ''
  items.value = []
  userProfile.value = null
  recommendDebug.value = 'started'
  const resolvedUserId = userId.value.trim() || authState.value?.recommendation_user_id || ''
  if (!resolvedUserId && !authState.value?.is_authenticated) {
    errorMessage.value = `missing_user_id auth=${Boolean(authState.value?.is_authenticated)}; bound=${authState.value?.recommendation_user_id || 'none'}`
    return
  }

  isRecommending.value = true
  const controller = new AbortController()
  const timeoutId = setTimeout(() => controller.abort(), 15000)

  try {
    const data = await fetchJson('/api/recommend/', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: resolvedUserId, top_k: 10 }),
      signal: controller.signal,
    })
    items.value = data.items || []
    recommendDebug.value = `fetched:${items.value.length}`
    const effectiveUserId = data.user_id || resolvedUserId || authState.value?.recommendation_user_id
    if (effectiveUserId) {
      userId.value = effectiveUserId
    }
    await Promise.all([effectiveUserId ? loadUserProfile(effectiveUserId) : Promise.resolve(), loadHistory(), loadDashboard()])
    recommendDebug.value = `synced:${items.value.length}`
    if (items.value.length) {
      await selectProduct(items.value[0].item_id)
    }
  } catch (error) {
    const message = error?.name === 'AbortError' ? 'recommend_timeout_15s' : (error?.message || 'unknown')
    recommendDebug.value = `error:${message}`
    try {
      const fallbackItems = await loadHistoryAndGetLatestResult(resolvedUserId)
      if (fallbackItems.length) {
        items.value = fallbackItems
        recommendDebug.value = `fallback:${items.value.length}`
        errorMessage.value = ''
        if (items.value.length) {
          await selectProduct(items.value[0].item_id)
        }
      } else {
        errorMessage.value = `recommend_error msg=${message}; debug=${recommendDebug.value}; bound=${authState.value?.recommendation_user_id || 'none'}; userId=${userId.value || 'empty'}; items=${items.value.length}`
      }
    } catch (fallbackError) {
      const fallbackMessage = fallbackError?.message || 'history_fallback_failed'
      errorMessage.value = `recommend_error msg=${message}; fallback=${fallbackMessage}; debug=${recommendDebug.value}; bound=${authState.value?.recommendation_user_id || 'none'}; userId=${userId.value || 'empty'}; items=${items.value.length}`
    }
  } finally {
    clearTimeout(timeoutId)
    isRecommending.value = false
  }
}

async function loadAll() {
  await Promise.all([loadDashboard(), loadAuthState()])
  if (authState.value?.recommendation_user_id) {
    userId.value = authState.value.recommendation_user_id
  }
}

onMounted(loadAll)
</script>

<style scoped>
.layout {
  display: grid;
  grid-template-columns: 1fr;
  grid-template-areas:
    'status'
    'recommend'
    'detail'
    'experiments'
    'history';
  gap: 20px;
  align-items: start;
}
.status-panel,
.recommend-panel,
.detail-panel,
.experiments-panel,
.history-catalog-panel {
  min-width: 0;
}
.status-panel { grid-area: status; }
.recommend-panel { grid-area: recommend; }
.detail-panel { grid-area: detail; }
.experiments-panel { grid-area: experiments; }
.history-catalog-panel { grid-area: history; }
</style>
