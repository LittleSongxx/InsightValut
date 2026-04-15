export interface AgenticEvidenceCoverageSummary {
  enabled?: boolean;
  coverage_score?: number;
  doc_count?: number;
  needs_rescue?: boolean;
  missing_focus_terms?: string[];
  missing_item_names?: string[];
  missing_sub_queries?: string[];
  clarification_required?: boolean;
  clarification_reason?: string;
}

export interface AgenticRescuePlan {
  action?: string;
  reason?: string;
  steps?: string[];
  coverage_score?: number;
}

export interface AgenticContextExpansionSummary {
  enabled?: boolean;
  candidate_docs?: number;
  expanded_docs?: number;
  affected_chunk_ids?: string[];
}

export interface AgenticAnswerPlan {
  query_type?: string;
  structured_output?: boolean;
  response_format?: string;
  sections?: string[];
  style_instructions?: string[];
  must_cover?: string[];
}

export interface AgenticMetadata {
  query_type?: string;
  graph_preferred?: boolean;
  query_focus_terms?: string[];
  query_route_reason?: string;
  retrieval_plan?: Record<string, unknown>;
  kg_query_summary?: Record<string, unknown>;
  evidence_coverage_summary?: AgenticEvidenceCoverageSummary;
  rescue_plan?: AgenticRescuePlan;
  context_expansion_summary?: AgenticContextExpansionSummary;
  answer_plan?: AgenticAnswerPlan;
  clarification_reason?: string;
  image_urls?: string[];
  agentic_features?: Record<string, boolean>;
}

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
  metadata?: AgenticMetadata;
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
  metadata?: AgenticMetadata | null;
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

export interface EvaluationMetricDelta {
  current: number | null;
  baseline: number | null;
  delta: number | null;
  relative_pct: number | null;
}

export interface EvaluationStageSummary {
  stage: string;
  avg_duration_ms: number | null;
  p95_duration_ms: number | null;
  count: number;
}

export interface EvaluationRetrievalGroundTruthSummary {
  eligible_cases: number;
  resolved_cases: number;
  unresolved_cases: number;
  source_breakdown: Record<string, number>;
  unresolved_reasons: Record<string, number>;
  stale_declared_cases: number;
}

export interface EvaluationSummary {
  variant: string;
  description: string;
  technique: string;
  case_count: number;
  ragas_metrics: Record<string, number | null>;
  ragas_coverage: Record<string, number>;
  ragas_errors: Record<string, string>;
  retrieval_metrics: Record<string, number | null>;
  retrieval_coverage: Record<string, number>;
  retrieval_ground_truth: EvaluationRetrievalGroundTruthSummary;
  pipeline_metrics: Record<string, number | null>;
  performance_metrics: {
    avg_total_duration_ms?: number | null;
    p50_total_duration_ms?: number | null;
    p95_total_duration_ms?: number | null;
    avg_first_answer_ms?: number | null;
    p50_first_answer_ms?: number | null;
    p95_first_answer_ms?: number | null;
    stages?: EvaluationStageSummary[];
      [key: string]: number | EvaluationStageSummary[] | null | undefined;
  };
  headline_metrics: Record<string, number | null>;
  warnings: string[];
}

export interface EvaluationVariantResult {
  description: string;
  technique: string;
  compare_to?: string;
  summary: EvaluationSummary;
  by_query_type: Record<string, EvaluationSummary>;
}

export interface EvaluationComparison {
  variant: string;
  compare_to: string;
  technique: string;
  overall: Record<string, EvaluationMetricDelta>;
  by_query_type: Record<string, Record<string, EvaluationMetricDelta>>;
}

export interface EvaluationReportListItem {
  report_id: string;
  file_name: string;
  generated_at: string;
  updated_at: string;
  dataset_path: string;
  dataset_name: string;
  case_count: number;
  final_variant: string;
  variants: string[];
  headline_metrics: Record<string, number | null>;
  size_bytes: number;
}

export interface EvaluationReportDetail {
  generated_at: string;
  dataset_path: string;
  case_count: number;
  final_variant: string;
  final_system_metrics: EvaluationSummary;
  variants: Record<string, EvaluationVariantResult>;
  comparisons: Record<string, EvaluationComparison>;
}

export interface EvaluationVariantOption {
  name: string;
  description: string;
  technique: string;
  compare_to?: string | null;
  is_default: boolean;
}

export interface EvaluationConfig {
  template_dataset_path: string;
  default_variants: string[];
  variant_catalog: EvaluationVariantOption[];
}

export interface EvaluationJob {
  job_id: string;
  status: 'pending' | 'running' | 'completed' | 'failed';
  dataset_path: string;
  variants: string[];
  output_path: string;
  progress_message: string;
  current_variant: string;
  completed_variants: number;
  total_variants: number;
  case_count: number;
  report_id: string;
  report_path: string;
  error: string;
  created_at: string;
  started_at: string;
  finished_at: string;
}
