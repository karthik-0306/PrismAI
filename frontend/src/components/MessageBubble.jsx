/**
 * src/components/MessageBubble.jsx
 *
 * Renders a single chat message. User messages are right-aligned.
 * Assistant messages are left-aligned with full Markdown rendering
 * (headings, bold, italic, tables, code blocks with syntax highlighting)
 * and rich metadata badges.
 */
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import styles from './MessageBubble.module.css';

/* ── Category configuration ─────────────────────────────────────────── */
const CATEGORY_CONFIG = {
  dsa:        { label: 'DSA',        icon: '🔗', color: 'purple'  },
  coding:     { label: 'Coding',     icon: '💻', color: 'cyan'    },
  math:       { label: 'Math',       icon: '📐', color: 'emerald' },
  science:    { label: 'Science',    icon: '🔬', color: 'amber'   },
  summarize:  { label: 'Summarize',  icon: '📋', color: 'rose'    },
  evaluate:   { label: 'Evaluate',   icon: '📊', color: 'amber'   },
  general:    { label: 'General',    icon: '💬', color: 'slate'   },
  fast:       { label: 'Fast',       icon: '⚡', color: 'violet'  },
  manual:     { label: 'Manual',     icon: '🎯', color: 'slate'   },
  web_search: { label: 'Web Search', icon: '🌐', color: 'sky'     },
};

/* ── Model display name shortener ───────────────────────────────────── */
function shortModelName(full) {
  if (!full || full === 'aggregated') return 'Aggregated';
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

/* ── Markdown code block renderer ────────────────────── */
// In react-markdown v9, 'inline' prop is gone.
// We detect inline vs block by: no className AND no newline in children = inline.
function CodeBlock({ children, className, node, ...props }) {
  const match = /language-(\w+)/.exec(className || '');
  const childStr = String(children);
  const isInline = !match && !childStr.includes('\n');

  if (isInline) {
    return (
      <code className={styles.inlineCode} {...props}>
        {children}
      </code>
    );
  }

  const language = match ? match[1] : 'text';
  const code = childStr.replace(/\n$/, '');

  return (
    <div className={styles.codeWrapper}>
      <div className={styles.codeHeader}>
        <span className={styles.codeLang}>{language}</span>
        <button
          className={styles.copyBtn}
          onClick={() => navigator.clipboard.writeText(code)}
          title="Copy code"
        >
          Copy
        </button>
      </div>
      <SyntaxHighlighter
        style={oneDark}
        language={language}
        PreTag="div"
        customStyle={{
          margin: 0,
          borderRadius: '0 0 8px 8px',
          fontSize: '13px',
          lineHeight: '1.6',
          background: '#0d0d1a',
        }}
        codeTagProps={{ style: { fontFamily: "'Fira Code', 'Cascadia Code', monospace" } }}
        {...props}
      >
        {code}
      </SyntaxHighlighter>
    </div>
  );
}

/* ── Markdown components map ────────────────────────────────────────── */
const markdownComponents = {
  code: CodeBlock,
  h1: ({ children }) => <h1 className={styles.mdH1}>{children}</h1>,
  h2: ({ children }) => <h2 className={styles.mdH2}>{children}</h2>,
  h3: ({ children }) => <h3 className={styles.mdH3}>{children}</h3>,
  h4: ({ children }) => <h4 className={styles.mdH4}>{children}</h4>,
  p:  ({ children }) => <p  className={styles.mdP}>{children}</p>,
  ul: ({ children }) => <ul className={styles.mdUl}>{children}</ul>,
  ol: ({ children }) => <ol className={styles.mdOl}>{children}</ol>,
  li: ({ children }) => <li className={styles.mdLi}>{children}</li>,
  strong: ({ children }) => <strong className={styles.mdStrong}>{children}</strong>,
  em: ({ children }) => <em className={styles.mdEm}>{children}</em>,
  blockquote: ({ children }) => <blockquote className={styles.mdBlockquote}>{children}</blockquote>,
  hr: () => <hr className={styles.mdHr} />,
  table: ({ children }) => (
    <div className={styles.tableWrapper}>
      <table className={styles.mdTable}>{children}</table>
    </div>
  ),
  th: ({ children }) => <th className={styles.mdTh}>{children}</th>,
  td: ({ children }) => <td className={styles.mdTd}>{children}</td>,
  a: ({ href, children }) => (
    <a className={styles.mdLink} href={href} target="_blank" rel="noreferrer">
      {children}
    </a>
  ),
};

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

/* ── Assistant Bubble ──────────────────────────────────────── */
function AssistantBubble({ content, categories_used, models_used, model_used, reduction_pct, isStreaming }) {
  const isAggregated = model_used === 'aggregated';

  return (
    <div className={styles.assistantRow}>
      {/* Avatar */}
      <div className={styles.avatar} aria-hidden="true">
        <span className={styles.avatarGlyph}>✦</span>
      </div>

      <div className={styles.assistantCard}>
        {/* Markdown-rendered response */}
        <div className={styles.prose}>
          <ReactMarkdown
            remarkPlugins={[remarkGfm]}
            components={markdownComponents}
          >
            {content}
          </ReactMarkdown>
          {/* Blinking cursor while streaming */}
          {isStreaming && <span className={styles.streamCursor} aria-hidden="true">▌</span>}
        </div>

        {/* Metadata badges row */}
        <div className={styles.metaRow} aria-label="Pipeline metadata">
          {/* Category badges */}
          {categories_used?.filter(c => c !== 'manual').map((cat) => {
            const cfg = CATEGORY_CONFIG[cat] || { label: cat, icon: '◆', color: 'slate' };
            return (
              <span key={cat} className={`${styles.badge} ${styles[`badge-${cfg.color}`]}`}>
                {cfg.label}
              </span>
            );
          })}

          {/* Model chip(s) */}
          {models_used?.length > 0 && !isAggregated && (
            <span className={styles.modelChip} title={models_used[0]}>
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
            <span className={styles.manualChip}>Manual</span>
          )}

          {/* Token savings chip */}
          {reduction_pct > 0.5 && (
            <span className={styles.savingsChip} title="Tokens saved by Rewriter">
              -{Number(reduction_pct).toFixed(1)}% tokens
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

/* ── Main Export ────────────────────────────────────────── */
export default function MessageBubble({ message, isStreaming }) {
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
      isStreaming={isStreaming}
    />
  );
}
