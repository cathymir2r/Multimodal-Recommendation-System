import { createApp } from 'vue'
import App from './App.vue'
import router from './router'

function fromCodePoints(points) {
  return String.fromCodePoint(...points)
}

const appTitle = fromCodePoints([22810, 27169, 24577, 21830, 21697, 25512, 33616, 31995, 32479])
const routeTitles = {
  dashboard: `${fromCodePoints([31995, 32479, 24635, 35272])} | ${appTitle}`,
  experiments: `${fromCodePoints([23454, 39564, 20998, 26512])} | ${appTitle}`,
  users: `${fromCodePoints([29992, 25143, 30011, 20687])} | ${appTitle}`,
  architecture: `${fromCodePoints([26550, 26500, 35828, 26126])} | ${appTitle}`,
  docs: `${fromCodePoints([25991, 26723, 23548, 20986])} | ${appTitle}`,
}

document.title = appTitle

router.afterEach((to) => {
  document.title = routeTitles[to.name] || appTitle
})

createApp(App).use(router).mount('#app')
