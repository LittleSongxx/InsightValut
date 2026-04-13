export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  isStreaming?: boolean;
  images?: string[];
  sources?: string[];
  startTime?: number;
  endTime?: number;
}

export interface ProgressData {
  status: string;
  done_list: string[];
  running_list: string[];
}

export interface HistoryItem {
  _id: string;
  session_id: string;
  role: string;
  text: string;
  rewritten_query: string;
  item_names: string[];
  image_urls: string[] | null;
  ts: number;
}

export interface ImportTask {
  task_id: string;
  file_name: string;
  file_title: string;
  item_name: string;
  status: string;
  done_list: string[];
  running_list: string[];
  error_message?: string;
  created_at: number;
  updated_at: number;
}

export interface PerformanceRecord {
  session_id: string;
  query: string;
  total_duration_ms: number;
  stages: StageRecord[];
  created_at: string;
}

export interface StageRecord {
  stage: string;
  duration_ms: number;
  status: 'success' | 'error';
  error?: string;
}

export interface PerformanceSummary {
  run_count: number;
  avg_total_duration_ms: number;
  p50_total_duration_ms: number;
  p95_total_duration_ms: number;
  avg_first_answer_ms: number | null;
  stages: StageBreakdown[];
}

export interface StageBreakdown {
  stage: string;
  count: number;
  avg_duration_ms: number;
  p95_duration_ms: number;
  error_rate: number;
}

export interface PerformanceTimePoint {
  period: string;
  run_count: number;
  avg_total_duration_ms: number;
  p95_total_duration_ms: number;
}
