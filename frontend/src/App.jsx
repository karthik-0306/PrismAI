/**
 * src/App.jsx
 *
 * Root component. Owns global state (chats, active chat, messages, settings)
 * and the send / stop / regenerate / retry flows.
 *
 * Messages are loaded from the server ONLY when the user selects a chat. The
 * active chat id also changes when the server creates a chat mid-stream, and
 * re-fetching then would overwrite the answer that is still streaming in.
 */
import { useState, useEffect, useCallback, useRef } from 'react';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import AnalyticsDashboard from './components/AnalyticsDashboard';
import ServerStatusBanner from './components/ServerStatusBanner';
import FreeTierNotice from './components/FreeTierNotice';
import { Toaster, toast } from 'react-hot-toast';
import { streamMessage, fetchChats, fetchMessages, deleteChat } from './api/chat';
import { useSession } from './hooks/useSession';
import { useBackendStatus } from './hooks/useBackendStatus';
import styles from './App.module.css';

const ACTIVE_CHAT_KEY = 'prismai_active_chat';

function readStoredChat() {
  try { return localStorage.getItem(ACTIVE_CHAT_KEY); } catch { return null; }
}
function storeChat(id) {
  try {
    if (id) localStorage.setItem(ACTIVE_CHAT_KEY, id);
    else localStorage.removeItem(ACTIVE_CHAT_KEY);
  } catch { /* storage unavailable — not critical */ }
}

function normaliseMessage(m) {
  return {
    ...m,
    _id: m.message_id,
    categories_used: m.categories_used ?? [],
    models_used: m.models_used ?? [],
    sources: m.sources ?? [],
  };
}

export default function App() {
  const sessionId = useSession();
  const backendStatus = useBackendStatus();

  const [chats, setChats] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [isStreaming, setIsStreaming] = useState(false);
  const [isLoadingChat, setIsLoadingChat] = useState(false);
  const [modelPreference, setModelPreference] = useState('auto');
  const [rewriterEnabled, setRewriterEnabled] = useState(true);
  const [currentView, setCurrentView] = useState('chat'); // 'chat' | 'analytics'

  const activeChatIdRef = useRef(null);
  const abortRef = useRef(null);
  const isStreamingRef = useRef(false);

  const setActiveChat = useCallback((id) => {
    activeChatIdRef.current = id;
    setActiveChatId(id);
    storeChat(id);
  }, []);

  const refreshChats = useCallback(() => (
    fetchChats(sessionId).then(setChats).catch((err) => console.error(err))
  ), [sessionId]);

  const loadChat = useCallback(async (chatId) => {
    setIsLoadingChat(true);
    try {
      const msgs = await fetchMessages(sessionId, chatId);
      if (activeChatIdRef.current === chatId) setMessages(msgs.map(normaliseMessage));
    } catch (err) {
      console.error(err);
      toast.error('Could not load this conversation');
    } finally {
      setIsLoadingChat(false);
    }
  }, [sessionId]);

  /* ── On mount: chat list + restore the last open chat ─────────── */
  useEffect(() => {
    fetchChats(sessionId)
      .then((list) => {
        setChats(list);
        const stored = readStoredChat();
        if (stored && list.some(c => c.chat_id === stored)) {
          setActiveChat(stored);
          loadChat(stored);
        }
      })
      .catch(console.error);
  }, [sessionId, setActiveChat, loadChat]);

  /* ── Navigation ──────────────────────────────────────────────── */
  const handleSelectChat = useCallback((chatId) => {
    if (isStreamingRef.current) abortRef.current?.abort();
    setCurrentView('chat');
    if (chatId === activeChatIdRef.current) return;
    setActiveChat(chatId);
    setMessages([]);
    loadChat(chatId);
  }, [setActiveChat, loadChat]);

  const handleNewChat = useCallback(() => {
    if (isStreamingRef.current) abortRef.current?.abort();
    setActiveChat(null);
    setMessages([]);
    setCurrentView('chat');
  }, [setActiveChat]);

  const handleViewAnalytics = useCallback(() => setCurrentView('analytics'), []);

  /* ── Send (also used for regenerate / retry) ─────────────────── */
  const runTurn = useCallback(async (text, { regenerate = false } = {}) => {
    if (!text.trim() || isStreamingRef.current) return;

    const asstId = `pending-${Date.now()}`;
    const userMsg = { _id: `user-${Date.now()}`, role: 'user', content: text, categories_used: [], models_used: [] };
    const placeholder = {
      _id: asstId, role: 'assistant', content: '', pending: true,
      status: 'Thinking…', categories_used: [], models_used: [], sources: [],
    };

    setMessages(prev => {
      let base = prev;
      if (regenerate) {
        // Drop the trailing assistant reply (or error) and the question it answered.
        base = [...prev];
        if (base.length && base[base.length - 1].role === 'assistant') base.pop();
        if (base.length && base[base.length - 1].role === 'user') base.pop();
      }
      return [...base, userMsg, placeholder];
    });

    const patch = (fn) => setMessages(prev => prev.map(m => (m._id === asstId ? { ...m, ...fn(m) } : m)));

    const controller = new AbortController();
    abortRef.current = controller;
    isStreamingRef.current = true;
    setIsStreaming(true);

    try {
      await streamMessage(
        {
          session_id: sessionId,
          chat_id: activeChatIdRef.current,
          message: text,
          model_preference: modelPreference,
          rewriter_enabled: rewriterEnabled,
          regenerate,
        },
        {
          onStart: (ev) => {
            if (ev.chat_id && ev.chat_id !== activeChatIdRef.current) {
              setActiveChat(ev.chat_id);
            }
            if (ev.is_new_chat) {
              const words = text.trim().split(/\s+/);
              const title = words.slice(0, 6).join(' ') + (words.length > 6 ? '…' : '');
              const now = new Date().toISOString();
              setChats(prev => [{ chat_id: ev.chat_id, title, created_at: now, updated_at: now },
                ...prev.filter(c => c.chat_id !== ev.chat_id)]);
            }
          },
          onStatus: (ev) => patch(() => ({ status: ev.label })),
          onSources: (sources) => patch(() => ({ sources })),
          onToken: (chunk) => patch(m => ({ content: m.content + chunk })),
          onReset: () => patch(() => ({ content: '' })),
          onDone: (meta) => {
            patch(() => ({
              _id: meta.message_id,
              message_id: meta.message_id,
              pending: false,
              status: null,
              model_used: meta.model_used,
              route_category: meta.route_category,
              categories_used: meta.categories_used ?? [],
              models_used: meta.models_used ?? [],
              original_tokens: meta.original_tokens ?? 0,
              rewritten_tokens: meta.rewritten_tokens ?? 0,
              reduction_pct: meta.reduction_pct ?? 0,
              sources: meta.sources ?? [],
            }));
            refreshChats();
          },
        },
        controller.signal,
      );

      if (controller.signal.aborted) {
        patch(m => ({ pending: false, status: null, stopped: true,
                      content: m.content || '_Response stopped._' }));
      }
    } catch (err) {
      console.error(err);
      patch(() => ({ pending: false, status: null, error: err.message || 'Something went wrong.' }));
      toast.error(err.message || 'Something went wrong');
    } finally {
      isStreamingRef.current = false;
      abortRef.current = null;
      setIsStreaming(false);
    }
  }, [sessionId, modelPreference, rewriterEnabled, setActiveChat, refreshChats]);

  const handleSend = useCallback((text) => runTurn(text), [runTurn]);

  const handleStop = useCallback(() => abortRef.current?.abort(), []);

  /* Regenerate the last answer, or retry a failed one — same server operation. */
  const handleRegenerate = useCallback(() => {
    const lastUser = [...messages].reverse().find(m => m.role === 'user');
    if (lastUser) runTurn(lastUser.content, { regenerate: true });
  }, [messages, runTurn]);

  /* ── Delete ──────────────────────────────────────────────────── */
  const handleDeleteChat = useCallback(async (chatId) => {
    try {
      await deleteChat(sessionId, chatId);
      setChats(prev => prev.filter(c => c.chat_id !== chatId));
      if (activeChatIdRef.current === chatId) handleNewChat();
      toast.success('Chat deleted');
    } catch (err) {
      console.error('Failed to delete chat:', err);
      toast.error('Failed to delete chat');
    }
  }, [sessionId, handleNewChat]);

  const activeChat = chats.find(c => c.chat_id === activeChatId);

  return (
    <div className={styles.layout}>
      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: 'var(--bg-card)',
            color: 'var(--text-primary)',
            border: '1px solid var(--border-subtle)',
            fontSize: '14px',
          },
        }}
      />
      <Sidebar
        chats={chats}
        activeChatId={activeChatId}
        onSelectChat={handleSelectChat}
        onNewChat={handleNewChat}
        onDeleteChat={handleDeleteChat}
        onViewAnalytics={handleViewAnalytics}
        modelPreference={modelPreference}
        onModelChange={setModelPreference}
        rewriterEnabled={rewriterEnabled}
        onRewriterToggle={() => setRewriterEnabled(v => !v)}
      />
      {currentView === 'analytics' ? (
        <AnalyticsDashboard />
      ) : (
        <div style={{ display: 'flex', flexDirection: 'column', flex: 1, overflow: 'hidden' }}>
          <ServerStatusBanner status={backendStatus} />
          <ChatArea
            messages={messages}
            isStreaming={isStreaming}
            isLoading={isLoadingChat}
            onSend={handleSend}
            onStop={handleStop}
            onRegenerate={handleRegenerate}
            activeChatTitle={activeChat?.title}
          />
        </div>
      )}
      <FreeTierNotice />
    </div>
  );
}
