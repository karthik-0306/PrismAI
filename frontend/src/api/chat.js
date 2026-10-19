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

async function errorFrom(res, fallback) {
  const body = await res.json().catch(() => null);
  const detail = body?.detail;
  if (Array.isArray(detail)) return new Error(detail[0]?.msg || fallback);
  return new Error(detail || `${fallback} (HTTP ${res.status})`);
}

/**
 * Stream a chat message via Server-Sent Events.
 *
 * Every query type streams — there is no non-streaming fallback anymore.
 * Handlers (all optional except onToken):
 *   onStart({chat_id, user_message_id, is_new_chat})  chat id is known immediately
 *   onStatus({stage, label})                          progress hint ("Searching the web…")
 *   onSources(sources)                                [{title, url}] for web answers
 *   onToken(chunk)
 *   onReset()                                         discard streamed text; a fallback model restarts
 *   onDone(metadata)
 * Rejects with an Error for HTTP errors and pipeline `error` events.
 * Pass an AbortSignal to support a Stop button (resolves quietly when aborted).
 */
export async function streamMessage(
  { session_id, chat_id, message, model_preference, rewriter_enabled, regenerate },
  { onStart, onStatus, onSources, onToken, onReset, onDone },
  signal,
) {
  let res;
  try {
    res = await fetch(`${BASE}/chat/stream`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id,
        chat_id: chat_id ?? null,
        message,
        model_preference: model_preference ?? 'auto',
        rewriter_enabled: rewriter_enabled ?? true,
        regenerate: Boolean(regenerate),
      }),
      signal,
    });
  } catch (err) {
    if (err.name === 'AbortError') return;
    throw new Error('Cannot reach the server. Check your connection and try again.', { cause: err });
  }

  if (!res.ok) throw await errorFrom(res, 'Request failed');

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop(); // last partial line stays in the buffer

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6).trim();
        if (raw === '[DONE]') return;

        let event;
        try { event = JSON.parse(raw); } catch { continue; }

        switch (event.type) {
          case 'start':    onStart?.(event); break;
          case 'status':   onStatus?.(event); break;
          case 'sources':  onSources?.(event.sources || []); break;
          case 'token':    onToken(event.content); break;
          case 'reset':    onReset?.(); break;
          case 'metadata': onDone?.(event); break;
          case 'error': {
            const err = new Error(event.detail || 'Something went wrong.');
            err.chatId = event.chat_id;
            throw err;
          }
          default: break;
        }
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') return;
    throw err;
  }
}

/** Non-streaming variant (same pipeline). */
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
  if (!res.ok) throw await errorFrom(res, 'Request failed');
  return res.json();
}

/** Models available for manual selection: [{value, label, sub}] */
export async function fetchModels() {
  const res = await fetch(`${BASE}/models`);
  if (!res.ok) throw await errorFrom(res, 'Failed to fetch models');
  return (await res.json()).models;
}

/** Fetch all chats for a session (sidebar list), most recently active first */
export async function fetchChats(session_id) {
  const res = await fetch(`${BASE}/chats?session_id=${encodeURIComponent(session_id)}`);
  if (!res.ok) throw await errorFrom(res, 'Failed to fetch chats');
  return res.json();
}

export async function searchChats(session_id, q) {
  const res = await fetch(`${BASE}/chats/search?session_id=${encodeURIComponent(session_id)}&q=${encodeURIComponent(q)}`);
  if (!res.ok) throw await errorFrom(res, 'Failed to search chats');
  return res.json();
}

/** Fetch all messages for a chat. Resolves to [] if the chat no longer exists. */
export async function fetchMessages(session_id, chat_id) {
  const res = await fetch(`${BASE}/chats/${chat_id}/messages?session_id=${encodeURIComponent(session_id)}`);
  if (res.status === 404) return [];
  if (!res.ok) throw await errorFrom(res, 'Failed to fetch messages');
  return res.json();
}

export async function renameChat(session_id, chat_id, title) {
  const res = await fetch(`${BASE}/chats/${chat_id}?session_id=${encodeURIComponent(session_id)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title }),
  });
  if (!res.ok) throw await errorFrom(res, 'Failed to rename chat');
  return res.json();
}

/** Delete a chat and all its messages */
export async function deleteChat(session_id, chat_id) {
  const res = await fetch(`${BASE}/chats/${chat_id}?session_id=${encodeURIComponent(session_id)}`, { method: 'DELETE' });
  if (!res.ok && res.status !== 404) throw await errorFrom(res, 'Failed to delete chat');
  return true;
}
