import React, { useState, useEffect } from 'react';
import { fetchModelStatus } from '../api/metrics';
import styles from './SystemHealth.module.css';

export default function SystemHealth() {
  const [statuses, setStatuses] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const data = await fetchModelStatus();
        setStatuses(data.models);
      } catch (err) {
        console.error("Failed to load model statuses", err);
      } finally {
        setLoading(false);
      }
    }
    load();
    // Poll every 30 seconds
    const interval = setInterval(load, 30000);
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
          <div key={s.model} className={styles.dotWrapper} title={`${s.name}: ${s.latency_ms > 0 ? s.latency_ms + 'ms' : 'Offline'}`}>
            <span className={`${styles.dot} ${styles['dot-' + s.status]}`} />
          </div>
        ))}
      </div>
    </div>
  );
}
