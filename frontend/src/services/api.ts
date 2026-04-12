import type { HistoryItem, PerformanceSummary, PerformanceTimePoint, StageBreakdown } from '../types';

const QUERY_BASE = '/api/query';
const IMPORT_BASE = '/api/import';

// ─── Query Service ───────────────────────────────────────────
export async function sendQuery(query: string, sessionId: string, isStream: boolean) {
  const res = await fetch(`${QUERY_BASE}/query`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query, session_id: sessionId, is_stream: isStream }),
  });
  if (!res.ok) throw new Error(`Query failed: ${res.status}`);
  return res.json();
}

export function createSSEConnection(sessionId: string): EventSource {
  return new EventSource(`${QUERY_BASE}/stream/${sessionId}`);
}

export async function getHistory(sessionId: string, limit = 50): Promise<{ session_id: string; items: HistoryItem[] }> {
  const res = await fetch(`${QUERY_BASE}/history/${sessionId}?limit=${limit}`);
  if (!res.ok) throw new Error(`History failed: ${res.status}`);
  return res.json();
}

export async function clearHistory(sessionId: string) {
  const res = await fetch(`${QUERY_BASE}/history/${sessionId}`, { method: 'DELETE' });
  if (!res.ok) throw new Error(`Clear history failed: ${res.status}`);
  return res.json();
}

// ─── Import Service ──────────────────────────────────────────
export async function uploadFile(file: File, onProgress?: (pct: number) => void): Promise<{ task_id: string; filename: string }> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const formData = new FormData();
    formData.append('files', file);

    xhr.upload.addEventListener('progress', (e) => {
      if (e.lengthComputable && onProgress) {
        onProgress(Math.round((e.loaded / e.total) * 100));
      }
    });

    xhr.addEventListener('load', () => {
      if (xhr.status >= 200 && xhr.status < 300) {
        try {
          const data = JSON.parse(xhr.responseText);
          const taskId = data.task_ids?.[0] ?? data.task_id;
          resolve({ task_id: taskId, filename: file.name });
        } catch {
          reject(new Error('Invalid response from server'));
        }
      } else {
        reject(new Error(`Upload failed (${xhr.status}): ${xhr.responseText}`));
      }
    });

    xhr.addEventListener('error', () => reject(new Error('Upload network error')));
    xhr.open('POST', `${IMPORT_BASE}/upload`);
    xhr.send(formData);
  });
}

export async function getImportStatus(taskId: string) {
  const res = await fetch(`${IMPORT_BASE}/status/${taskId}`);
  if (!res.ok) throw new Error(`Status failed: ${res.status}`);
  return res.json();
}

export async function getImportList() {
  const res = await fetch(`${IMPORT_BASE}/tasks`);
  if (!res.ok) throw new Error(`Task list failed: ${res.status}`);
  return res.json();
}

export interface SessionInfo {
  session_id: string;
  title: string;
  last_ts: number;
  message_count: number;
}

export async function getSessions(limit = 50): Promise<SessionInfo[]> {
  const res = await fetch(`${QUERY_BASE}/sessions?limit=${limit}`);
  if (!res.ok) throw new Error(`Sessions failed: ${res.status}`);
  const data = await res.json();
  return data.sessions || [];
}

// ─── Performance Service ─────────────────────────────────────
export async function getPerformanceSummaryData(
  startDate?: string,
  endDate?: string,
): Promise<PerformanceSummary> {
  const params = new URLSearchParams();
  if (startDate) params.append('start_date', startDate);
  if (endDate) params.append('end_date', endDate);
  const query = params.toString();
  const res = await fetch(`${QUERY_BASE}/performance/summary${query ? '?' + query : ''}`);
  if (!res.ok) throw new Error(`Performance summary failed: ${res.status}`);
  return res.json();
}

export async function getPerformanceTimeSeries(
  granularity: 'day' | 'hour' = 'day',
  startDate?: string,
  endDate?: string,
): Promise<PerformanceTimePoint[]> {
  const params = new URLSearchParams({ granularity });
  if (startDate) params.append('start_date', startDate);
  if (endDate) params.append('end_date', endDate);
  const res = await fetch(`${QUERY_BASE}/performance/time-series?${params.toString()}`);
  if (!res.ok) throw new Error(`Performance time-series failed: ${res.status}`);
  return res.json();
}

export async function getStageBreakdown(
  startDate?: string,
  endDate?: string,
): Promise<StageBreakdown[]> {
  const params = new URLSearchParams();
  if (startDate) params.append('start_date', startDate);
  if (endDate) params.append('end_date', endDate);
  const query = params.toString();
  const res = await fetch(`${QUERY_BASE}/performance/stages${query ? '?' + query : ''}`);
  if (!res.ok) throw new Error(`Stage breakdown failed: ${res.status}`);
  return res.json();
}
