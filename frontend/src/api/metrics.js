/**
 * src/api/metrics.js
 * Fetches analytics data and model status from the backend.
 */

const BASE = import.meta.env.VITE_API_BASE_URL ?? '/api';

export async function fetchMetrics(session_id) {
  const res = await fetch(`${BASE}/metrics?session_id=${session_id}`);
  if (!res.ok) throw new Error(`Failed to fetch metrics: HTTP ${res.status}`);
  const ct = res.headers.get('content-type') ?? '';
  if (!ct.includes('application/json')) {
    throw new Error('Backend is not reachable — make sure the server is running');
  }
  return res.json();
}

export async function fetchModelStatus() {
  const res = await fetch(`${BASE}/model-status`);
  if (!res.ok) throw new Error(`Failed to fetch model status: HTTP ${res.status}`);
  const ct = res.headers.get('content-type') ?? '';
  if (!ct.includes('application/json')) {
    throw new Error('Backend is not reachable — make sure the server is running');
  }
  return res.json();
}
