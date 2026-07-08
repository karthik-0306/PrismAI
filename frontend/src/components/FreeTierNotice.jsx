/**
 * src/components/FreeTierNotice.jsx
 *
 * A persistent, non-dismissable sticky notice that stays permanently
 * at the bottom of the screen informing users about free-tier limitations
 * and pointing them to the local setup.
 */

export default function FreeTierNotice() {
  return (
    <div
      id="free-tier-notice"
      style={{
        position: 'fixed',
        bottom: 0,
        left: 0,
        right: 0,
        zIndex: 9999,
        background: 'linear-gradient(90deg, rgba(251,146,60,0.10) 0%, rgba(251,191,36,0.10) 100%)',
        borderTop: '1px solid rgba(251,146,60,0.30)',
        backdropFilter: 'blur(8px)',
        WebkitBackdropFilter: 'blur(8px)',
        padding: '7px 16px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: '10px',
        flexWrap: 'wrap',
        fontFamily: 'inherit',
        fontSize: '12px',
        color: 'rgba(251,191,36,0.92)',
        userSelect: 'none',
      }}
    >
      <span style={{ textAlign: 'center', lineHeight: 1.5 }}>
        <strong style={{ color: '#fbbf24' }}>Free-tier notice:</strong>
        {' '}The backend is deployed on Render's free tier — it may be slow, crash, or timeout occasionally.
        {' '}
        <a
          href="https://github.com/karthik-0306/PrismAI"
          target="_blank"
          rel="noopener noreferrer"
          style={{
            color: '#fb923c',
            textDecoration: 'underline',
            fontWeight: 600,
            cursor: 'pointer',
          }}
        >
          Clone &amp; run locally
        </a>
        {' '}for the best experience.
      </span>
    </div>
  );
}
