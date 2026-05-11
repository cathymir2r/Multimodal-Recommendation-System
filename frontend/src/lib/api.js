export async function fetchJson(url, options = undefined) {
  const response = await fetch(url, { credentials: 'include', ...(options || {}) })
  const contentType = response.headers.get('content-type') || ''

  let data
  if (contentType.includes('application/json')) {
    data = await response.json()
  } else {
    const rawText = await response.text()
    throw new Error(rawText.startsWith('<!DOCTYPE') ? 'non_json_html_response' : (rawText || 'empty_non_json_response'))
  }

  if (!response.ok) {
    throw new Error(data.detail || `http_${response.status}`)
  }
  return data
}

export function formatMetric(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return '-'
  }
  return Number(value).toFixed(4)
}

export function formatDate(value) {
  if (!value) {
    return '-'
  }
  return new Date(value).toLocaleString('zh-CN')
}

export const placeholderImage = "data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='300' height='300'><rect width='100%25' height='100%25' fill='%23e5e7eb'/><text x='50%25' y='50%25' dominant-baseline='middle' text-anchor='middle' fill='%236b7280' font-size='18'>No Image</text></svg>"
