/**
 * src/App.jsx
 *
 * Root component. Manages global state: active chat, messages, model preference.
 * Wires Sidebar ↔ ChatArea with clean prop drilling.
 */
import { useState, useEffect, useCallback, useRef } from 'react';
import Sidebar from './components/Sidebar';
import ChatArea from './components/ChatArea';
import AnalyticsDashboard from './components/AnalyticsDashboard';
import ServerStatusBanner from './components/ServerStatusBanner';
import { Toaster, toast } from 'react-hot-toast';
import { streamMessage, sendMessage, fetchChats, fetchMessages } from './api/chat';
import { useSession } from './hooks/useSession';
import { useBackendStatus } from './hooks/useBackendStatus';
import styles from './App.module.css';

export default function App() {
  const sessionId = useSession();
  const backendStatus = useBackendStatus();

  const [chats, setChats] = useState([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [isThinking, setIsThinking] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [modelPreference, setModelPreference] = useState('auto');
  const [rewriterEnabled, setRewriterEnabled] = useState(true);
  const [currentView, setCurrentView] = useState('chat'); // 'chat' or 'analytics'
  const activeChatIdRef = useRef(null);

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
    activeChatIdRef.current = chatId;
    setCurrentView('chat');
  }, []);

  /* ── Start a fresh chat ──────────────────────────────────────── */
  const handleNewChat = useCallback(() => {
    setActiveChatId(null);
    activeChatIdRef.current = null;
    setMessages([]);
    setCurrentView('chat');
  }, []);

  /* ── View Analytics ──────────────────────────────────────────── */
  const handleViewAnalytics = useCallback(() => {
    setCurrentView('analytics');
  }, []);

  /* ── Send a message ──────────────────────────────────────────── */
  const handleSend = useCallback(async (text) => {
    if (!text.trim() || isThinking || isStreaming) return;

    // Optimistically append user message
    const tempUserId = `temp-user-${Date.now()}`;
    const tempAsstId = `temp-asst-${Date.now()}`;
    const userMsg = { _id: tempUserId, role: 'user', content: text, categories_used: [], models_used: [] };

    // Placeholder assistant bubble — will be filled token by token
    const asstPlaceholder = { _id: tempAsstId, role: 'assistant', content: '', categories_used: [], models_used: [] };

    setMessages(prev => [...prev, userMsg, asstPlaceholder]);
    setIsStreaming(true);

    const currentChatId = activeChatIdRef.current;

    try {
      await streamMessage(
        { session_id: sessionId, chat_id: currentChatId, message: text, model_preference: modelPreference, rewriter_enabled: rewriterEnabled },
        {
          onToken: (chunk) => {
            setMessages(prev => prev.map(m =>
              m._id === tempAsstId ? { ...m, content: m.content + chunk } : m
            ));
          },
          onDone: async (meta) => {
            // Merge final metadata into the assistant bubble
            setMessages(prev => prev.map(m =>
              m._id === tempAsstId ? {
                ...m,
                _id: meta.message_id,
                message_id: meta.message_id,
                model_used: meta.model_used,
                route_category: meta.route_category,
                categories_used: meta.categories_used ?? [],
                models_used: meta.models_used ?? [],
                original_tokens: meta.original_tokens ?? 0,
                rewritten_tokens: meta.rewritten_tokens ?? 0,
                reduction_pct: meta.reduction_pct ?? 0,
              } : m
            ));

            // Update active chat id + sidebar for new chats
            if (!currentChatId && meta.chat_id) {
              setActiveChatId(meta.chat_id);
              activeChatIdRef.current = meta.chat_id;
              const updated = await fetchChats(sessionId).catch(() => chats);
              setChats(updated);
            }
          },
          onFallback: async (fallbackChatId) => {
            // Compound query — remove placeholder, show ThinkingBubble, use non-streaming path
            setMessages(prev => prev.filter(m => m._id !== tempAsstId));
            setIsStreaming(false);
            setIsThinking(true);

            const resolvedChatId = fallbackChatId || currentChatId;
            try {
              const data = await sendMessage({
                session_id: sessionId,
                chat_id: resolvedChatId,
                message: text,
                model_preference: modelPreference,
                rewriter_enabled: rewriterEnabled,
              });

              if (!currentChatId && data.chat_id) {
                setActiveChatId(data.chat_id);
                activeChatIdRef.current = data.chat_id;
                const updated = await fetchChats(sessionId).catch(() => chats);
                setChats(updated);
              }

              setMessages(prev => [...prev, {
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
              }]);
            } finally {
              setIsThinking(false);
            }
          },
        }
      );
    } catch (err) {
      console.error(err);
      setMessages(prev => prev.map(m =>
        m._id === tempAsstId ? { ...m, content: '⚠️ Error generating response.' } : m
      ));
      toast.error(err.message || 'Something went wrong');
    } finally {
      setIsStreaming(false);
    }
  }, [sessionId, activeChatIdRef, modelPreference, rewriterEnabled, isThinking, isStreaming, chats]);

  /* ── Delete a chat ───────────────────────────────────────────── */
  const handleDeleteChat = useCallback(async (chatId) => {
    try {
      // Import here to avoid circular dependencies if any, or use the top-level import
      const { deleteChat } = await import('./api/chat.js');
      await deleteChat(chatId);
      
      // Update local state
      setChats(prev => prev.filter(c => c.chat_id !== chatId));
      
      // If the deleted chat was currently open, clear the screen
      if (activeChatIdRef.current === chatId) {
        handleNewChat();
      }
      
      toast.success('Chat deleted');
    } catch (err) {
      console.error('Failed to delete chat:', err);
      toast.error('Failed to delete chat');
    }
  }, [handleNewChat]);

  /* ── Derive active chat title ────────────────────────────────── */
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
          }
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
            isThinking={isThinking}
            isStreaming={isStreaming}
            onSend={handleSend}
            activeChatTitle={activeChat?.title}
          />
        </div>
      )}
    </div>
  );
}
