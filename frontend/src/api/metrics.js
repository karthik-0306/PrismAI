/**
 * src/api/metrics.js
 * Fetches analytics data and model status from the backend.
 */

const BASE = '/api';

export async function fetchMetrics(session_id) {
  const res = await fetch(`${BASE}/metrics?session_id=${session_id}`);
  if (!res.ok) throw new Error(`Failed to fetch metrics: HTTP ${res.status}`);
  return res.json();
}

export async function fetchModelStatus() {
  const res = await fetch(`${BASE}/model-status`);
  if (!res.ok) throw new Error(`Failed to fetch model status: HTTP ${res.status}`);
  return res.json();
}
