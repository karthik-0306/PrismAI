/**
 * src/api/chat.js
 *
 * All HTTP calls to the FastAPI backend.
 *
 * In development: requests go to /api and Vite's proxy forwards them to localhost:8000
 * In production (Vercel): VITE_API_BASE_URL is set to the Render backend URL
 *   e.g. https://prismai-backend.onrender.com/api
 */

const BASE = import.meta.env.VITE_API_BASE_URL ?? '/api';

/**
 * Stream a chat message via Server-Sent Events.
 *
 * @param {object} params  - Same fields as sendMessage.
 * @param {function} onToken   - Called with each text chunk: onToken(chunk: string)
 * @param {function} onDone    - Called once with the final metadata object
 * @param {function} onFallback - Called when a compound query requires non-streaming fallback
 */
export async function streamMessage(
  { session_id, chat_id, message, model_preference, rewriter_enabled },
  { onToken, onDone, onFallback }
) {
  const res = await fetch(`${BASE}/chat/stream`, {
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

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop(); // last partial line stays in buffer

    for (const line of lines) {
      if (!line.startsWith('data: ')) continue;
      const raw = line.slice(6).trim();
      if (raw === '[DONE]') return;

      let event;
      try { event = JSON.parse(raw); } catch { continue; }

      if (event.type === 'token') {
        onToken(event.content);
      } else if (event.type === 'metadata') {
        onDone(event);
      } else if (event.type === 'fallback') {
        // Compound query — hand off to non-streaming path
        await onFallback(event.chat_id);
      } else if (event.type === 'error') {
        throw new Error(event.detail || 'Streaming error');
      }
    }
  }
}

/** Non-streaming fallback — used for compound queries and history loads */
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

export async function searchChats(session_id, q) {
  const res = await fetch(`${BASE}/chats/search?session_id=${session_id}&q=${encodeURIComponent(q)}`);
  if (!res.ok) throw new Error('Failed to search chats');
  return res.json();
}

/** Fetch all messages for a specific chat */
export async function fetchMessages(chat_id) {
  const res = await fetch(`${BASE}/chats/${chat_id}/messages`);
  if (!res.ok) throw new Error(`Failed to fetch messages: HTTP ${res.status}`);
  return res.json();
}

/** Delete a chat and all its messages */
export async function deleteChat(chat_id) {
  const res = await fetch(`${BASE}/chats/${chat_id}`, { method: 'DELETE' });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Failed to delete chat' }));
    throw new Error(err.detail || `HTTP ${res.status}`);
  }
  return res.json();
}
