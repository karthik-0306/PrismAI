import React, { useState, useEffect } from 'react';
import { fetchModelStatus } from '../api/metrics';
import styles from './SystemHealth.module.css';

// The backend reports status from real traffic (no LLM calls per poll), so a
// relaxed interval is plenty.
const POLL_INTERVAL_MS = 60 * 1000;

const STATUS_COLOR = { green: 'green', yellow: 'yellow', red: 'red', unknown: 'yellow' };

export default function SystemHealth() {
  const [statuses, setStatuses] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const data = await fetchModelStatus();
        setStatuses(data.models);
      } catch (err) {
        console.error('Failed to load model statuses', err);
      } finally {
        setLoading(false);
      }
    }
    load();
    const interval = setInterval(load, POLL_INTERVAL_MS);
    return () => clearInterval(interval);
  }, []);

  if (loading && statuses.length === 0) {
    return (
      <div className={styles.healthContainer}>
        <span className={styles.label}>System Health:</span>
        <span className={styles.loading}>Checking...</span>
      </div>
    );
  }

  return (
    <div className={styles.healthContainer}>
      <span className={styles.label}>System Health:</span>
      <div className={styles.dotsRow}>
        {statuses.map(s => (
          <div
            key={s.model}
            className={styles.dotWrapper}
            title={`${s.name}: ${s.latency_ms > 0 ? `${s.latency_ms}ms` : s.status === 'red' ? 'Unavailable' : 'No recent traffic'}`}
          >
            <span className={`${styles.dot} ${styles['dot-' + (STATUS_COLOR[s.status] || 'yellow')]}`} />
          </div>
        ))}
      </div>
    </div>
  );
}
