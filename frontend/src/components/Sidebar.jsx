/**
 * src/components/Sidebar.jsx
 *
 * Left panel: logo, new chat button, list of past conversations.
 */
import { useState, useEffect } from 'react';
import { useSession } from '../hooks/useSession';
import { searchChats, fetchModels } from '../api/chat';
import styles from './Sidebar.module.css';

// Manual-mode models come from the backend (GET /api/models) so they can never
// drift from what the server actually runs.
const AUTO_OPTION = { value: 'auto', label: 'Auto Route', sub: 'Smart multi-agent routing' };

function PrismIcon() {
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none" aria-hidden="true">
      <defs>
        <linearGradient id="prism-grad" x1="50%" y1="100%" x2="50%" y2="0%">
          <stop offset="0%" stopColor="#a78bfa"/>
          <stop offset="100%" stopColor="#67e8f9"/>
        </linearGradient>
      </defs>
      <polygon points="14,2 26,22 2,22" fill="url(#prism-grad)" opacity="0.9"/>
      <polygon points="14,8 22,22 6,22" fill="rgba(255,255,255,0.12)"/>
    </svg>
  );
}

function formatDate(dateStr) {
  if (!dateStr) return '';
  // Server timestamps are UTC "YYYY-MM-DD HH:MM:SS" without a zone marker.
  const iso = dateStr.includes('T') ? dateStr : dateStr.replace(' ', 'T') + 'Z';
  const d = new Date(iso);
  const now = new Date();
  const diffMs = now - d;
  const diffDays = Math.floor(diffMs / 86400000);
  if (diffDays === 0) return 'Today';
  if (diffDays === 1) return 'Yesterday';
  if (diffDays < 7)  return `${diffDays}d ago`;
  return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
}

export default function Sidebar({
  chats,
  activeChatId,
  onSelectChat,
  onNewChat,
  modelPreference,
  onModelChange,
  rewriterEnabled,
  onRewriterToggle,
  onViewAnalytics,
  onDeleteChat,
}) {
  const sessionId = useSession();
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [isSearching, setIsSearching] = useState(false);
  const [modelOptions, setModelOptions] = useState([AUTO_OPTION]);

  useEffect(() => {
    fetchModels()
      .then((models) => setModelOptions([AUTO_OPTION, ...models]))
      .catch((err) => console.error('Failed to load models:', err));
  }, []);

  // If the selected model disappears from the server list, fall back to auto.
  useEffect(() => {
    if (modelOptions.length > 1 && !modelOptions.some(o => o.value === modelPreference)) {
      onModelChange('auto');
    }
  }, [modelOptions, modelPreference, onModelChange]);

  useEffect(() => {
    if (!searchQuery.trim()) {
      setSearchResults([]);
      setIsSearching(false);
      return;
    }

    const timer = setTimeout(async () => {
      setIsSearching(true);
      try {
        const results = await searchChats(sessionId, searchQuery);
        setSearchResults(results);
      } catch (err) {
        console.error("Search failed:", err);
      } finally {
        setIsSearching(false);
      }
    }, 400);

    return () => clearTimeout(timer);
  }, [searchQuery, sessionId]);

  const displayedChats = searchQuery.trim() ? searchResults : chats;

  return (
    <aside className={styles.sidebar} aria-label="Sidebar">
      {/* Logo */}
      <div className={styles.logo}>
        <PrismIcon />
        <div>
          <span className={styles.logoName}>PrismAI</span>
          <span className={styles.logoTagline}>Multi-Agent Intelligence</span>
        </div>
      </div>

      {/* New Chat */}
      <button
        id="btn-new-chat"
        className={styles.newChatBtn}
        style={{ marginBottom: '8px' }}
        onClick={onNewChat}
        aria-label="Start new chat"
      >
        <span className={styles.newChatIcon}>+</span>
        New Chat
      </button>

      {/* Analytics Dashboard */}
      <button
        id="btn-analytics"
        className={styles.analyticsBtn}
        onClick={onViewAnalytics}
        aria-label="View Analytics Dashboard"
      >
        Analytics Dashboard
      </button>

      {/* Model Selector */}
      <div className={styles.section}>
        <p className={styles.sectionLabel}>Model</p>
        <div className={styles.selectWrap}>
          <select
            id="model-selector"
            className={styles.select}
            value={modelPreference}
            onChange={(e) => onModelChange(e.target.value)}
            aria-label="Select model"
          >
            {modelOptions.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
          <span className={styles.selectArrow}>▾</span>
        </div>
        <p className={styles.modelSub}>
          {modelOptions.find(o => o.value === modelPreference)?.sub}
        </p>
      </div>

      {/* Rewriter toggle */}
      <div className={styles.toggleRow}>
        <div>
          <p className={styles.toggleLabel}>Query Rewriter</p>
          <p className={styles.toggleSub}>Optimize query before dispatch</p>
        </div>
        <button
          id="btn-rewriter-toggle"
          className={`${styles.toggle} ${rewriterEnabled ? styles.toggleOn : ''}`}
          onClick={onRewriterToggle}
          role="switch"
          aria-checked={rewriterEnabled}
          aria-label="Toggle query rewriter"
        >
          <span className={styles.toggleThumb} />
        </button>
      </div>

      {/* Chat history */}
      <div className={styles.historyHeader}>
        <span className={styles.sectionLabel}>Conversations</span>
        <span className={styles.chatCount}>{displayedChats.length}</span>
      </div>

      <div className={styles.searchWrap}>
        <input
          type="text"
          className={styles.searchInput}
          placeholder="Search chats..."
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          aria-label="Search conversations"
        />
      </div>

      <nav className={styles.chatList} aria-label="Chat history">
        {isSearching ? (
          <p className={styles.emptyChatMsg}>Searching...</p>
        ) : displayedChats.length === 0 ? (
          <p className={styles.emptyChatMsg}>
            {searchQuery.trim() ? "No results found" : "No conversations yet"}
          </p>
        ) : null}

        {!isSearching && displayedChats.map((chat) => (
          <div key={chat.chat_id} className={styles.chatItemWrapper}>
            <button
              id={`chat-${chat.chat_id}`}
              className={`${styles.chatItem} ${chat.chat_id === activeChatId ? styles.chatItemActive : ''}`}
              onClick={() => {
                onSelectChat(chat.chat_id);
                setSearchQuery(''); // clear search on selection
              }}
              aria-current={chat.chat_id === activeChatId ? 'page' : undefined}
            >
              <span className={styles.chatTitle}>{chat.title}</span>
              <span className={styles.chatDate}>{formatDate(chat.updated_at || chat.created_at)}</span>
            </button>
            <button
              className={styles.deleteBtn}
              onClick={(e) => {
                e.stopPropagation();
                if (window.confirm('Delete this chat?')) {
                  onDeleteChat(chat.chat_id);
                }
              }}
              aria-label="Delete chat"
            >
              <svg width="12" height="12" viewBox="0 0 12 12" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" aria-hidden="true">
                <line x1="2" y1="2" x2="10" y2="10"/><line x1="10" y1="2" x2="2" y2="10"/>
              </svg>
            </button>
          </div>
        ))}
      </nav>

    </aside>
  );
}
