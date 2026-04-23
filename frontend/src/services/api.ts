import type {
  ChunkIdMigrationResult,
  EvaluationConfig,
  EvaluationDatasetSyncResult,
  EvaluationJob,
  EvaluationReportDetail,
  EvaluationReportDeleteResult,
  EvaluationReportListItem,
  EvaluationVariantTrialResult,
  HistoryItem,
  ImportTask,
  PerformanceSummary,
  PerformanceTimePoint,
  QueryCacheResetResult,
  QueryCacheStats,
  StageBreakdown,
} from '../types';

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

export interface ImportStatusResponse {
  code: number;
  task_id: string;
  status: string;
  done_list: string[];
  running_list: string[];
  error_message: string;
  item_name: string;
  file_name: string;
  file_title: string;
}

export async function getImportStatus(taskId: string): Promise<ImportStatusResponse> {
  const res = await fetch(`${IMPORT_BASE}/status/${taskId}`);
  if (!res.ok) throw new Error(`Status failed: ${res.status}`);
  return res.json();
}

export async function getImportList(limit = 100, status?: string): Promise<{ code: number; tasks: ImportTask[] }> {
  const params = new URLSearchParams({ limit: String(limit) });
  if (status) params.append('status', status);
  const res = await fetch(`${IMPORT_BASE}/tasks?${params.toString()}`);
  if (!res.ok) throw new Error(`Task list failed: ${res.status}`);
  return res.json();
}

export interface DeleteImportTasksResponse {
  code: number;
  requested_count: number;
  deleted_count: number;
  deleted_task_ids: string[];
  skipped: Array<{ task_id: string; reason: string }>;
}

export async function deleteImportTasks(taskIds: string[]): Promise<DeleteImportTasksResponse> {
  const res = await fetch(`${IMPORT_BASE}/tasks/delete`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ task_ids: taskIds }),
  });
  if (!res.ok) throw new Error(`Delete tasks failed: ${res.status}`);
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

export async function getQueryCacheStats(): Promise<QueryCacheStats> {
  const res = await fetch(`${QUERY_BASE}/cache/stats`);
  if (!res.ok) throw new Error(`Query cache stats failed: ${res.status}`);
  return res.json();
}

export async function resetQueryCache(reason = 'manual'): Promise<QueryCacheResetResult> {
  const res = await fetch(`${QUERY_BASE}/cache/reset`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ reason }),
  });
  if (!res.ok) {
    let message = `Reset query cache failed: ${res.status}`;
    try {
      const data = await res.json();
      if (data?.detail) message = data.detail;
    } catch {
      // noop
    }
    throw new Error(message);
  }
  return res.json();
}

// ─── Evaluation Service ──────────────────────────────────────
export async function getEvaluationReports(): Promise<{
  reports: EvaluationReportListItem[];
  latest_report_id: string | null;
}> {
  const res = await fetch(`${QUERY_BASE}/evaluation/reports`);
  if (!res.ok) throw new Error(`Evaluation reports failed: ${res.status}`);
  return res.json();
}

export async function getLatestEvaluationReport(): Promise<{
  report: { meta: EvaluationReportListItem; report: EvaluationReportDetail } | null;
}> {
  const res = await fetch(`${QUERY_BASE}/evaluation/reports/latest`);
  if (!res.ok) throw new Error(`Latest evaluation report failed: ${res.status}`);
  return res.json();
}

export async function getEvaluationReport(reportId: string): Promise<{
  meta: EvaluationReportListItem;
  report: EvaluationReportDetail;
}> {
  const res = await fetch(`${QUERY_BASE}/evaluation/reports/${encodeURIComponent(reportId)}`);
  if (!res.ok) throw new Error(`Evaluation report failed: ${res.status}`);
  return res.json();
}

export async function deleteEvaluationReport(reportId: string): Promise<EvaluationReportDeleteResult> {
  const res = await fetch(`${QUERY_BASE}/evaluation/reports/${encodeURIComponent(reportId)}`, {
    method: 'DELETE',
  });
  if (!res.ok) {
    let message = `Delete evaluation report failed: ${res.status}`;
    try {
      const data = await res.json();
      if (data?.detail) message = data.detail;
    } catch {
      // noop
    }
    throw new Error(message);
  }
  return res.json();
}

export async function getEvaluationConfig(): Promise<EvaluationConfig> {
  const res = await fetch(`${QUERY_BASE}/evaluation/config`);
  if (!res.ok) throw new Error(`Evaluation config failed: ${res.status}`);
  return res.json();
}

export async function getEvaluationJobs(limit = 10): Promise<{ jobs: EvaluationJob[] }> {
  const res = await fetch(`${QUERY_BASE}/evaluation/jobs?limit=${limit}`);
  if (!res.ok) throw new Error(`Evaluation jobs failed: ${res.status}`);
  return res.json();
}

export async function getEvaluationJob(jobId: string): Promise<EvaluationJob> {
  const res = await fetch(`${QUERY_BASE}/evaluation/jobs/${encodeURIComponent(jobId)}`);
  if (!res.ok) throw new Error(`Evaluation job failed: ${res.status}`);
  return res.json();
}

export async function cancelEvaluationJob(jobId: string): Promise<EvaluationJob> {
  const res = await fetch(`${QUERY_BASE}/evaluation/jobs/${encodeURIComponent(jobId)}/cancel`, {
    method: 'POST',
  });
  if (!res.ok) {
    let message = `Cancel evaluation job failed: ${res.status}`;
    try {
      const data = await res.json();
      if (data?.detail) message = data.detail;
    } catch {
      // noop
    }
    throw new Error(message);
  }
  return res.json();
}

export async function createEvaluationJob(payload: {
  dataset_path: string;
  variants: string[];
  output_path?: string;
}): Promise<EvaluationJob> {
  const res = await fetch(`${QUERY_BASE}/evaluation/jobs`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    let message = `Create evaluation job failed: ${res.status}`;
    try {
      const data = await res.json();
      if (data?.detail) message = data.detail;
    } catch {
      // noop
    }
    throw new Error(message);
  }
  return res.json();
}

export async function testEvaluationVariant(payload: {
  query: string;
  variant_name: string;
}): Promise<EvaluationVariantTrialResult> {
  const res = await fetch(`${QUERY_BASE}/evaluation/variants/test`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    let message = `Evaluation variant test failed: ${res.status}`;
    try {
      const data = await res.json();
      if (data?.detail) message = data.detail;
    } catch {
      // noop
    }
    throw new Error(message);
  }
  return res.json();
}

export async function syncEvaluationDatasetChunkIds(payload: {
  dataset_path: string;
  output_path?: string;
  create_backup?: boolean;
}): Promise<EvaluationDatasetSyncResult> {
  const res = await fetch(`${QUERY_BASE}/evaluation/dataset/sync`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload),
  });
  if (!res.ok) {
    let message = `Sync evaluation dataset failed: ${res.status}`;
    try {
      const data = await res.json();
      if (data?.detail) message = data.detail;
    } catch {
      // noop
    }
    throw new Error(message);
  }
  return res.json();
}

export async function migrateKnowledgeBaseChunkIds(payload?: {
  item_names?: string[];
  collection_name?: string;
  dry_run?: boolean;
  sync_graph?: boolean;
}): Promise<ChunkIdMigrationResult> {
  const res = await fetch(`${QUERY_BASE}/knowledge-base/chunk-ids/migrate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload || {}),
  });
  if (!res.ok) {
    let message = `Migrate chunk ids failed: ${res.status}`;
    try {
      const data = await res.json();
      if (data?.detail) message = data.detail;
    } catch {
      // noop
    }
    throw new Error(message);
  }
  return res.json();
}
