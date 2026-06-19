/**
 * src/components/Sidebar.jsx
 *
 * Left panel: logo, new chat button, list of past conversations.
 */
import styles from './Sidebar.module.css';

const MODEL_OPTIONS = [
  { value: 'auto',                        label: '✦ Auto Route',       sub: 'Smart multi-agent routing'  },
  { value: 'groq/openai/gpt-oss-120b',    label: 'GPT-OSS 120B',       sub: 'Most powerful · Groq'        },
  { value: 'groq/qwen/qwen3-32b',         label: 'Qwen3 32B',          sub: 'Strong reasoning · Groq'     },
  { value: 'groq/llama-3.3-70b-versatile',label: 'Llama 3.3 70B',      sub: 'Balanced · Groq'             },
  { value: 'gemini/gemini-3.5-flash',     label: 'Gemini 3.5 Flash',   sub: 'Fast · Google'               },
  { value: 'groq/llama-3.1-8b-instant',   label: 'Llama 3.1 8B',       sub: 'Fastest · Groq'              },
];

function PrismIcon() {
  return (
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none" aria-hidden="true">
      <defs>
        <linearGradient id="prism-grad" x1="0%" y1="0%" x2="100%" y2="100%">
          <stop offset="0%" stopColor="#7c3aed"/>
          <stop offset="50%" stopColor="#06b6d4"/>
          <stop offset="100%" stopColor="#10b981"/>
        </linearGradient>
      </defs>
      <polygon points="14,2 26,22 2,22" fill="url(#prism-grad)" opacity="0.9"/>
      <polygon points="14,8 22,22 6,22" fill="rgba(255,255,255,0.12)"/>
    </svg>
  );
}

function formatDate(dateStr) {
  const d = new Date(dateStr);
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
}) {
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
        onClick={onNewChat}
        aria-label="Start new chat"
      >
        <span className={styles.newChatIcon}>+</span>
        New Chat
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
            {MODEL_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
          <span className={styles.selectArrow}>▾</span>
        </div>
        <p className={styles.modelSub}>
          {MODEL_OPTIONS.find(o => o.value === modelPreference)?.sub}
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
        <span className={styles.chatCount}>{chats.length}</span>
      </div>

      <nav className={styles.chatList} aria-label="Chat history">
        {chats.length === 0 && (
          <p className={styles.emptyChatMsg}>No conversations yet</p>
        )}
        {chats.map((chat) => (
          <button
            key={chat.chat_id}
            id={`chat-${chat.chat_id}`}
            className={`${styles.chatItem} ${chat.chat_id === activeChatId ? styles.chatItemActive : ''}`}
            onClick={() => onSelectChat(chat.chat_id)}
            aria-current={chat.chat_id === activeChatId ? 'page' : undefined}
          >
            <span className={styles.chatDot} />
            <span className={styles.chatTitle}>{chat.title}</span>
            <span className={styles.chatDate}>{formatDate(chat.created_at)}</span>
          </button>
        ))}
      </nav>

      {/* Footer */}
      <div className={styles.footer}>
        <div className={styles.footerBadge}>
          <span className={styles.footerDot} />
          Phase 3 Live
        </div>
        <span className={styles.footerVersion}>v0.3.0</span>
      </div>
    </aside>
  );
}
