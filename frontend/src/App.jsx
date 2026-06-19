/**
 * src/App.jsx
 *
 * Root component. Manages global state: active chat, messages, model preference.
 * Wires Sidebar ↔ ChatArea with clean prop drilling.
 */
import { useState, useEffect, useCallback } from 'react';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import { sendMessage, fetchChats, fetchMessages } from './api/chat';
import { useSession } from './hooks/useSession';
import styles from './App.module.css';

export default function App() {
  const sessionId = useSession();

  const [chats, setChats] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [isThinking, setIsThinking] = useState(false);
  const [modelPreference, setModelPreference] = useState('auto');
  const [rewriterEnabled, setRewriterEnabled] = useState(true);

  /* ── Load chat list on mount ──────────────────────────────── */
  useEffect(() => {
    fetchChats(sessionId)
      .then(setChats)
      .catch(console.error);
  }, [sessionId]);

  /* ── Load messages when active chat changes ─────────────────── */
  useEffect(() => {
    if (!activeChatId) {
      setMessages([]);
      return;
    }
    fetchMessages(activeChatId)
      .then((msgs) => {
        // Normalise: history messages don't have categories_used / models_used;
        // we just render them without badges.
        setMessages(msgs.map(m => ({
          ...m,
          _id: m.message_id,
          categories_used: m.categories_used ?? [],
          models_used: m.models_used ?? [],
        })));
      })
      .catch(console.error);
  }, [activeChatId]);

  /* ── Select an existing chat ─────────────────────────────────── */
  const handleSelectChat = useCallback((chatId) => {
    setActiveChatId(chatId);
  }, []);

  /* ── Start a fresh chat ──────────────────────────────────────── */
  const handleNewChat = useCallback(() => {
    setActiveChatId(null);
    setMessages([]);
  }, []);

  /* ── Send a message ──────────────────────────────────────────── */
  const handleSend = useCallback(async (text) => {
    if (!text.trim() || isThinking) return;

    // Optimistically append user message
    const tempId = `temp-${Date.now()}`;
    const userMsg = {
      _id: tempId,
      role: 'user',
      content: text,
      categories_used: [],
      models_used: [],
    };
    setMessages(prev => [...prev, userMsg]);
    setIsThinking(true);

    try {
      const data = await sendMessage({
        session_id: sessionId,
        chat_id: activeChatId,
        message: text,
        model_preference: modelPreference,
        rewriter_enabled: rewriterEnabled,
      });

      // If this was a new chat, store the returned chat_id
      if (!activeChatId) {
        setActiveChatId(data.chat_id);
        // Refresh sidebar
        const updated = await fetchChats(sessionId).catch(() => chats);
        setChats(updated);
      }

      // Append the assistant response with full metadata
      const assistantMsg = {
        _id: data.message_id,
        message_id: data.message_id,
        role: 'assistant',
        content: data.response,
        model_used: data.model_used,
        route_category: data.route_category,
        categories_used: data.categories_used ?? [],
        models_used: data.models_used ?? [],
        original_tokens: data.original_tokens ?? 0,
        rewritten_tokens: data.rewritten_tokens ?? 0,
        reduction_pct: data.reduction_pct ?? 0,
      };
      setMessages(prev => [...prev, assistantMsg]);

    } catch (err) {
      console.error(err);
      // Show an error bubble
      setMessages(prev => [...prev, {
        _id: `err-${Date.now()}`,
        role: 'assistant',
        content: `⚠️ Something went wrong: ${err.message}`,
        categories_used: [],
        models_used: [],
      }]);
    } finally {
      setIsThinking(false);
    }
  }, [sessionId, activeChatId, modelPreference, rewriterEnabled, isThinking, chats]);

  /* ── Derive active chat title ────────────────────────────────── */
  const activeChat = chats.find(c => c.chat_id === activeChatId);

  return (
    <div className={styles.layout}>
      <Sidebar
        chats={chats}
        activeChatId={activeChatId}
        onSelectChat={handleSelectChat}
        onNewChat={handleNewChat}
        modelPreference={modelPreference}
        onModelChange={setModelPreference}
        rewriterEnabled={rewriterEnabled}
        onRewriterToggle={() => setRewriterEnabled(v => !v)}
      />
      <ChatArea
        messages={messages}
        isThinking={isThinking}
        onSend={handleSend}
        activeChatTitle={activeChat?.title}
      />
    </div>
  );
}
