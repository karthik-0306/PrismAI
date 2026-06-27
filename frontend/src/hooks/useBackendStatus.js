/**
 * src/hooks/useBackendStatus.js
 *
 * Pings the backend /health endpoint on mount.
 * Returns:
 *   'checking'  — initial state, request in flight
 *   'waking'    — backend didn't respond in time (cold start in progress)
 *   'ready'     — backend responded healthy
 *   'error'     — backend returned an error / unreachable
 */
import { useState, useEffect } from 'react';

const HEALTH_URL = (import.meta.env.VITE_API_BASE_URL ?? '/api')
  .replace(/\/api$/, '') + '/health';

const FAST_TIMEOUT_MS  = 4000;   // if no response in 4s → show wakeup banner
const POLL_INTERVAL_MS = 5000;   // retry every 5s until healthy

export function useBackendStatus() {
  const [status, setStatus] = useState('checking'); // 'checking' | 'waking' | 'ready' | 'error'

  useEffect(() => {
    let cancelled = false;
    let pollTimer = null;

    async function ping() {
      const controller = new AbortController();
      const timeout = setTimeout(() => controller.abort(), FAST_TIMEOUT_MS);

      try {
        const res = await fetch(HEALTH_URL, { signal: controller.signal });
        clearTimeout(timeout);
        if (cancelled) return;

        if (res.ok) {
          setStatus('ready');
        } else {
          setStatus('error');
        }
      } catch {
        clearTimeout(timeout);
        if (cancelled) return;
        // Aborted (timeout) or network error → server is waking up
        setStatus('waking');
        // Keep polling until it's alive
        pollTimer = setInterval(async () => {
          const pollController = new AbortController();
          const pollTimeoutId = setTimeout(() => pollController.abort(), 5000);
          
          try {
            const r = await fetch(HEALTH_URL, { signal: pollController.signal });
            clearTimeout(pollTimeoutId);
            if (!cancelled && r.ok) {
              setStatus('ready');
              clearInterval(pollTimer);
            }
          } catch {
            clearTimeout(pollTimeoutId);
            // Still waking up — keep polling
          }
        }, POLL_INTERVAL_MS);
      }
    }

    ping();

    return () => {
      cancelled = true;
      if (pollTimer) clearInterval(pollTimer);
    };
  }, []);

  return status;
}
