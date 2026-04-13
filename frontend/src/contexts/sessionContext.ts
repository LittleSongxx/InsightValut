import { createContext } from 'react';
import type { SessionInfo } from '../services/api';

export interface SessionContextValue {
  sessions: SessionInfo[];
  currentSessionId: string | null;
  sessionsLoaded: boolean;
  refreshSessions: () => Promise<SessionInfo[]>;
  selectSession: (sessionId: string | null) => void;
  createNewSession: () => string;
}

export const SessionContext = createContext<SessionContextValue>({
  sessions: [],
  currentSessionId: null,
  sessionsLoaded: false,
  refreshSessions: async () => [],
  selectSession: () => {},
  createNewSession: () => '',
});
