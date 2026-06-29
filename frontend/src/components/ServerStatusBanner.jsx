/**
 * src/components/ServerStatusBanner.jsx
 *
 * Shows a contextual banner based on backend connectivity:
 *   - 'checking'  → subtle "Connecting to backend..." pulse
 *   - 'waking'    → prominent "Server warming up (free tier)" with spinner and ETA
 *   - 'ready'     → brief "Server ready ✓" green flash, then disappears
 *   - 'error'     → red warning
 */
import { useEffect, useState } from 'react';
import styles from './ServerStatusBanner.module.css';

export default function ServerStatusBanner({ status }) {
  const [visible, setVisible] = useState(true);
  const [elapsed, setElapsed] = useState(0);

  // Auto-hide after a short delay once ready
  useEffect(() => {
    if (status === 'ready') {
      const t = setTimeout(() => setVisible(false), 2500);
      return () => clearTimeout(t);
    }
    setVisible(true);
  }, [status]);

  // Elapsed seconds counter shown during wakeup
  useEffect(() => {
    if (status !== 'waking') { setElapsed(0); return; }
    const t = setInterval(() => setElapsed(s => s + 1), 1000);
    return () => clearInterval(t);
  }, [status]);

  if (!visible || status === 'checking') return null;

  const config = {
    waking: {
      icon: null,
      cls: styles.waking,
      title: 'Backend server is warming up',
      sub: `Free-tier cold start — usually takes 30–60 seconds. (${elapsed}s elapsed)`,
    },
    ready: {
      icon: '✓',
      cls: styles.ready,
      title: 'Server is ready',
      sub: 'All systems operational.',
    },
    error: {
      icon: null,
      cls: styles.error,
      title: 'Cannot reach backend',
      sub: 'Please refresh the page or try again in a moment.',
    },
  }[status];

  if (!config) return null;

  return (
    <div className={`${styles.banner} ${config.cls}`} role="status">
      <span className={styles.icon}>{config.icon}</span>
      <div className={styles.text}>
        <span className={styles.title}>{config.title}</span>
        <span className={styles.sub}>{config.sub}</span>
      </div>
      {status === 'waking' && <span className={styles.spinner} aria-hidden="true" />}
    </div>
  );
}
