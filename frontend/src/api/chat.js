/**
 * src/api/chat.js
 *
 * All HTTP calls to the FastAPI backend.
 * Uses a relative URL so that in development, Vite's proxy handles CORS.
 */

const BASE = '/api';

/** Send a chat message and stream back the response */
export async function sendMessage({ session_id, chat_id, message, model_preference, rewriter_enabled }) {
  const res = await fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      session_id,
      chat_id: chat_id ?? null,
      message,
      model_preference: model_preference ?? 'auto',
      rewriter_enabled: rewriter_enabled ?? true,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Unknown error' }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }

  return res.json();
}

/** Fetch all chats for a session (sidebar list) */
export async function fetchChats(session_id) {
  const res = await fetch(`${BASE}/chats?session_id=${session_id}`);
  if (!res.ok) throw new Error(`Failed to fetch chats: HTTP ${res.status}`);
  return res.json();
}

/** Fetch all messages for a specific chat */
export async function fetchMessages(chat_id) {
  const res = await fetch(`${BASE}/chats/${chat_id}/messages`);
  if (!res.ok) throw new Error(`Failed to fetch messages: HTTP ${res.status}`);
  return res.json();
}
