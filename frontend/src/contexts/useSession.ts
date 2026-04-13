import { useContext } from 'react';
import { SessionContext } from './sessionContext';

export function useSession() {
  return useContext(SessionContext);
}
