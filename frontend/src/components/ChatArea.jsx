/**
 * src/components/ChatArea.jsx
 *
 * The main chat window: header, scrollable message list, input bar.
 */
import { useRef, useEffect } from 'react';
import MessageBubble, { ThinkingBubble } from './MessageBubble';
import styles from './ChatArea.module.css';

function SendIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <line x1="22" y1="2" x2="11" y2="13"/>
      <polygon points="22 2 15 22 11 13 2 9 22 2"/>
    </svg>
  );
}

function EmptyState() {
  return (
    <div className={styles.empty}>
      <div className={styles.emptyGlow} aria-hidden="true" />
      <div className={styles.emptyPrism} aria-hidden="true">
        <svg width="56" height="56" viewBox="0 0 56 56" fill="none">
          <defs>
            <linearGradient id="eg" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#7c3aed"/>
              <stop offset="50%" stopColor="#06b6d4"/>
              <stop offset="100%" stopColor="#10b981"/>
            </linearGradient>
          </defs>
          <polygon points="28,4 52,44 4,44" fill="url(#eg)" opacity="0.85"/>
          <polygon points="28,16 44,44 12,44" fill="rgba(255,255,255,0.1)"/>
        </svg>
      </div>
      <h1 className={styles.emptyTitle}>PrismAI</h1>
      <p className={styles.emptySubtitle}>
        Multi-agent intelligence that routes your questions<br/>to the best specialized model automatically.
      </p>
      <div className={styles.emptyChips}>
        {['🔗 DSA &amp; Algorithms', '📐 Mathematics', '💻 Coding', '🔬 Science', '📋 Summarize'].map(c => (
          <span key={c} className={styles.emptyChip} dangerouslySetInnerHTML={{ __html: c }} />
        ))}
      </div>
    </div>
  );
}

export default function ChatArea({ messages, isThinking, isStreaming, onSend, activeChatTitle }) {
  const listRef = useRef(null);
  const inputRef = useRef(null);

  /* Auto-scroll when new messages arrive */
  useEffect(() => {
    if (listRef.current) {
      listRef.current.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' });
    }
  }, [messages, isThinking]);

  function handleKeyDown(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  }

  /* Calculate total session token savings */
  const totalOriginal = messages.reduce((sum, m) => sum + (m.original_tokens || 0), 0);
  const totalRewritten = messages.reduce((sum, m) => sum + (m.rewritten_tokens || 0), 0);
  const savedTokens = totalOriginal - totalRewritten;
  const savingsPct = totalOriginal > 0 && savedTokens > 0 ? Math.round((savedTokens / totalOriginal) * 100) : 0;

  function submit() {
    const text = inputRef.current?.value?.trim();
    if (!text || isThinking) return;
    inputRef.current.value = '';
    onSend(text);
  }

  function handleExport() {
    if (messages.length === 0) return;
    
    let md = `# ${activeChatTitle || 'Chat Export'}\n\n`;
    messages.forEach(m => {
      md += `### ${m.role === 'user' ? 'You' : 'PrismAI'}\n`;
      md += `${m.content}\n\n---\n\n`;
    });

    const blob = new Blob([md], { type: 'text/markdown' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${(activeChatTitle || 'chat').replace(/[^a-z0-9]/gi, '_').toLowerCase()}.md`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  }

  return (
    <main className={styles.chatArea} aria-label="Chat area">
      {/* Header */}
      <header className={styles.header}>
        <div className={styles.headerLeft}>
          <span className={styles.headerIndicator} aria-hidden="true" />
          <h2 className={styles.headerTitle}>{activeChatTitle || 'New Chat'}</h2>
        </div>
        <div className={styles.headerRight}>
          {messages.length > 0 && (
            <button className={styles.exportBtn} onClick={handleExport} aria-label="Export chat to Markdown">
              📥 Export
            </button>
          )}
          {savingsPct > 0 && (
            <span className={styles.headerSavings} title={`${savedTokens} tokens saved across session`}>
              ⚡ {savingsPct}% compression
            </span>
          )}
          <span className={styles.headerTag}>Phase 3 · Multi-Agent</span>
        </div>
      </header>

      {/* Messages */}
      <div className={styles.messageList} ref={listRef} role="log" aria-live="polite" aria-label="Conversation">
        {messages.length === 0 ? (
          <EmptyState />
        ) : (
          messages.map((msg, idx) => {
            const isLastAsst = msg.role === 'assistant' && idx === messages.length - 1;
            return (
              <MessageBubble
                key={msg.message_id || msg._id}
                message={msg}
                isStreaming={isStreaming && isLastAsst}
              />
            );
          })
        )}
        {isThinking && <ThinkingBubble />}
      </div>

      {/* Input bar */}
      <div className={styles.inputBar}>
        <div className={styles.inputWrap}>
          <textarea
            id="chat-input"
            ref={inputRef}
            className={styles.input}
            placeholder="Ask anything — PrismAI routes it to the right model…"
            rows={1}
            onKeyDown={handleKeyDown}
            onInput={(e) => {
              e.target.style.height = 'auto';
              e.target.style.height = Math.min(e.target.scrollHeight, 140) + 'px';
            }}
            aria-label="Message input"
            disabled={isThinking || isStreaming}
          />
          <button
            id="btn-send"
            className={styles.sendBtn}
            onClick={submit}
            disabled={isThinking || isStreaming}
            aria-label="Send message"
          >
            {isThinking ? (
              <span className={styles.sendSpinner} aria-hidden="true" />
            ) : (
              <SendIcon />
            )}
          </button>
        </div>
        <p className={styles.inputHint}>
          <kbd>Enter</kbd> to send · <kbd>Shift+Enter</kbd> for new line
        </p>
      </div>
    </main>
  );
}
