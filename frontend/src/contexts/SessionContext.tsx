import { useCallback, useEffect, useMemo, useState } from 'react';
import type { ReactNode } from 'react';
import { getSessions } from '../services/api';
import type { SessionInfo } from '../services/api';
import { SessionContext } from './sessionContext';

const SESSION_STORAGE_KEY = 'iv_session_id';

function generateSessionId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 9);
}

export function SessionProvider({ children }: { children: ReactNode }) {
  const [sessions, setSessions] = useState<SessionInfo[]>([]);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(() => localStorage.getItem(SESSION_STORAGE_KEY));
  const [draftSessionId, setDraftSessionId] = useState<string | null>(null);
  const [sessionsLoaded, setSessionsLoaded] = useState(false);

  const persistSessionId = useCallback((sessionId: string | null) => {
    if (sessionId) {
      localStorage.setItem(SESSION_STORAGE_KEY, sessionId);
    } else {
      localStorage.removeItem(SESSION_STORAGE_KEY);
    }
    setCurrentSessionId(sessionId);
  }, []);

  const refreshSessions = useCallback(async () => {
    const list = await getSessions(200);
    setSessions(list);
    setSessionsLoaded(true);
    return list;
  }, []);

  useEffect(() => {
    const timerId = window.setTimeout(() => {
      void refreshSessions().catch(() => undefined);
    }, 0);

    return () => {
      window.clearTimeout(timerId);
    };
  }, [refreshSessions]);

  useEffect(() => {
    if (!sessionsLoaded) return;

    if (draftSessionId && sessions.some((session) => session.session_id === draftSessionId)) {
      queueMicrotask(() => {
        setDraftSessionId(null);
      });
      return;
    }

    if (draftSessionId && currentSessionId === draftSessionId) {
      return;
    }

    if (currentSessionId && sessions.some((session) => session.session_id === currentSessionId)) {
      return;
    }

    const storedSessionId = localStorage.getItem(SESSION_STORAGE_KEY);
    if (storedSessionId && sessions.some((session) => session.session_id === storedSessionId)) {
      queueMicrotask(() => {
        setCurrentSessionId(storedSessionId);
      });
      return;
    }

    if (sessions.length > 0) {
      queueMicrotask(() => {
        persistSessionId(sessions[0].session_id);
      });
      return;
    }

    if (currentSessionId) {
      queueMicrotask(() => {
        persistSessionId(null);
      });
    }
  }, [currentSessionId, draftSessionId, persistSessionId, sessions, sessionsLoaded]);

  const selectSession = useCallback((sessionId: string | null) => {
    setDraftSessionId(null);
    persistSessionId(sessionId);
  }, [persistSessionId]);

  const createNewSession = useCallback(() => {
    const nextSessionId = generateSessionId();
    setDraftSessionId(nextSessionId);
    persistSessionId(nextSessionId);
    return nextSessionId;
  }, [persistSessionId]);

  const value = useMemo(() => ({
    sessions,
    currentSessionId,
    sessionsLoaded,
    refreshSessions,
    selectSession,
    createNewSession,
  }), [createNewSession, currentSessionId, refreshSessions, selectSession, sessions, sessionsLoaded]);

  return <SessionContext.Provider value={value}>{children}</SessionContext.Provider>;
}
