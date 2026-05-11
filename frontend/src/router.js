import { createRouter, createWebHistory } from 'vue-router'

import DashboardView from './views/DashboardView.vue'
import ExperimentsView from './views/ExperimentsView.vue'
import UsersView from './views/UsersView.vue'
import ArchitectureView from './views/ArchitectureView.vue'
import DocsView from './views/DocsView.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', name: 'dashboard', component: DashboardView },
    { path: '/experiments', name: 'experiments', component: ExperimentsView },
    { path: '/users', name: 'users', component: UsersView },
    { path: '/architecture', name: 'architecture', component: ArchitectureView },
    { path: '/docs', name: 'docs', component: DocsView },
  ],
})

export default router
