/**
 * src/hooks/useSession.js
 *
 * Generates and persists a UUID4 session ID in localStorage.
 * Every browser gets one fixed session ID that identifies all its chats.
 */
import { useState } from 'react';

function generateUUID() {
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, (c) => {
    const r = Math.random() * 16 | 0;
    return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
  });
}

export function useSession() {
  const [sessionId] = useState(() => {
    let id = localStorage.getItem('prismai_session_id');
    if (!id) {
      id = generateUUID();
      localStorage.setItem('prismai_session_id', id);
    }
    return id;
  });
  return sessionId;
}
