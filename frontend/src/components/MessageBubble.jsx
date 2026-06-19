/**
 * src/components/MessageBubble.jsx
 *
 * Renders a single chat message. User messages are right-aligned.
 * Assistant messages are left-aligned with rich metadata badges showing
 * which categories were used and which models actually answered.
 */
import styles from './MessageBubble.module.css';

/* ── Category configuration ─────────────────────────────────────────── */
const CATEGORY_CONFIG = {
  dsa:      { label: 'DSA',       icon: '🔗', color: 'purple' },
  coding:   { label: 'Coding',    icon: '💻', color: 'cyan'   },
  math:     { label: 'Math',      icon: '📐', color: 'emerald'},
  science:  { label: 'Science',   icon: '🔬', color: 'amber'  },
  summarize:{ label: 'Summarize', icon: '📋', color: 'rose'   },
  general:  { label: 'General',   icon: '💬', color: 'slate'  },
  fast:     { label: 'Fast',      icon: '⚡', color: 'violet' },
  manual:   { label: 'Manual',    icon: '🎯', color: 'slate'  },
};

/* ── Model display name shortener ───────────────────────────────────── */
function shortModelName(full) {
  if (!full || full === 'aggregated') return 'Aggregated';
  // "groq/openai/gpt-oss-120b" → "gpt-oss-120b"
  // "gemini/gemini-3.5-flash"  → "Gemini 3.5 Flash"
  // "groq/llama-3.1-8b-instant"→ "Llama 3.1 8B"
  const part = full.split('/').pop();
  return part
    .replace(/-instant$/, '')
    .replace(/-versatile$/, '')
    .replace(/(\d)b$/, (_, n) => `${n}B`)
    .split('-').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
}

/* ── Timestamp ──────────────────────────────────────────────────────── */
function now() {
  return new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

/* ── User Bubble ─────────────────────────────────────────────────────── */
function UserBubble({ content }) {
  return (
    <div className={styles.userRow}>
      <div className={styles.userBubble}>
        <p className={styles.content}>{content}</p>
        <span className={styles.time}>{now()}</span>
      </div>
    </div>
  );
}

/* ── Assistant Bubble ────────────────────────────────────────────────── */
function AssistantBubble({ content, categories_used, models_used, model_used, reduction_pct }) {
  const isAggregated = model_used === 'aggregated';

  return (
    <div className={styles.assistantRow}>
      {/* Avatar */}
      <div className={styles.avatar} aria-hidden="true">
        <span className={styles.avatarGlyph}>✦</span>
      </div>

      <div className={styles.assistantCard}>
        {/* Response text */}
        <div className={styles.prose}>
          {content.split('\n').map((line, i) => (
            line.trim() === '' ? <br key={i} /> : <p key={i}>{line}</p>
          ))}
        </div>

        {/* Metadata badges row */}
        <div className={styles.metaRow} aria-label="Pipeline metadata">
          {/* Category badges */}
          {categories_used?.filter(c => c !== 'manual').map((cat) => {
            const cfg = CATEGORY_CONFIG[cat] || { label: cat, icon: '◆', color: 'slate' };
            return (
              <span key={cat} className={`${styles.badge} ${styles[`badge-${cfg.color}`]}`}>
                <span className={styles.badgeIcon}>{cfg.icon}</span>
                {cfg.label}
              </span>
            );
          })}

          {/* Model chip(s) */}
          {models_used?.length > 0 && !isAggregated && (
            <span className={styles.modelChip} title={models_used[0]}>
              <span className={styles.chipDot} />
              {shortModelName(models_used[0])}
            </span>
          )}

          {/* Aggregated chip */}
          {isAggregated && (
            <span className={styles.aggregateChip}>
              <svg width="10" height="10" viewBox="0 0 10 10" fill="none" aria-hidden="true">
                <circle cx="2" cy="5" r="1.5" fill="currentColor"/>
                <circle cx="5" cy="2" r="1.5" fill="currentColor"/>
                <circle cx="8" cy="5" r="1.5" fill="currentColor"/>
                <circle cx="5" cy="8" r="1.5" fill="currentColor"/>
              </svg>
              Aggregated · {models_used?.length} models
            </span>
          )}

          {/* Manual chip */}
          {categories_used?.includes('manual') && (
            <span className={styles.manualChip}>🎯 Manual</span>
          )}

          {/* Token savings chip */}
          {reduction_pct > 0 && (
            <span className={styles.savingsChip} title="Tokens saved by Rewriter">
              ✂️ -{reduction_pct}% tokens
            </span>
          )}

          <span className={styles.time}>{now()}</span>
        </div>
      </div>
    </div>
  );
}

/* ── Thinking / Loading Bubble ──────────────────────────────────────── */
export function ThinkingBubble() {
  return (
    <div className={styles.assistantRow}>
      <div className={styles.avatar}>
        <span className={styles.avatarGlyph}>✦</span>
      </div>
      <div className={styles.thinkingCard} aria-label="AI is thinking">
        <div className={styles.dots}>
          <span /><span /><span />
        </div>
        <span className={styles.thinkingLabel}>Thinking…</span>
      </div>
    </div>
  );
}

/* ── Main Export ─────────────────────────────────────────────────────── */
export default function MessageBubble({ message }) {
  if (message.role === 'user') {
    return <UserBubble content={message.content} />;
  }
  return (
    <AssistantBubble
      content={message.content}
      categories_used={message.categories_used}
      models_used={message.models_used}
      model_used={message.model_used}
      reduction_pct={message.reduction_pct}
    />
  );
}
