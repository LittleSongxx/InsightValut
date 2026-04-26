import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Activity,
  BarChart3,
  CheckCircle2,
  Clock3,
  Database,
  FileText,
  Loader2,
  Pencil,
  Play,
  Plus,
  RefreshCcw,
  Search,
  ShieldAlert,
  Square,
  Sparkles,
  Target,
  Trash2,
  X,
} from 'lucide-react';
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  appendEvaluationReport,
  cancelEvaluationJob,
  createEvaluationJob,
  deleteEvaluationReport,
  getEvaluationConfig,
  getEvaluationJob,
  getEvaluationJobs,
  getEvaluationReport,
  getEvaluationReports,
  getPerformanceSummaryData,
  getPerformanceTimeSeries,
  getQueryCacheStats,
  resetQueryCache,
  getStageBreakdown,
  migrateKnowledgeBaseChunkIds,
  syncEvaluationDatasetChunkIds,
  testEvaluationVariant,
} from '../services/api';
import { MarkdownRenderer } from '../components/chat/MarkdownRenderer';
import type {
  ChunkIdMigrationResult,
  EvaluationConfig,
  EvaluationDatasetSyncResult,
  EvaluationFeatureOption,
  EvaluationFeatureVariantSpec,
  EvaluationJob,
  EvaluationMetricDelta,
  EvaluationReportDetail,
  EvaluationReportListItem,
  EvaluationRetrievalGroundTruthSummary,
  EvaluationSummary,
  EvaluationVariantOption,
  EvaluationVariantTrialResult,
  PerformanceSummary,
  PerformanceTimePoint,
  QueryCacheStats,
  StageBreakdown,
} from '../types';

const STAGE_LABELS: Record<string, string> = {
  node_item_name_confirm: '确认问题产品',
  node_answer_output: '生成答案',
  node_rerank: '重排序',
  node_rrf: '倒排融合',
  node_search_embedding: '切片搜索',
  node_search_embedding_hyde: '切片搜索(HyDE)',
  node_multi_search: '多路搜索',
  node_query_kg: '查询知识图谱',
  node_join: '多路搜索合并',
  node_web_search_mcp: '网络搜索',
  node_query_decompose: '复合问题分解',
  node_retrieval_grader: '检索质量评估',
  node_hallucination_check: '幻觉自检',
};

const QUERY_TYPE_LABELS: Record<string, string> = {
  general: '通用问答',
  navigation: '导航定位',
  comparison: '对比问答',
  relation: '关系问答',
  constraint: '约束问答',
  explain: '解释问答',
};

const METRIC_LABELS: Record<string, string> = {
  factual_correctness: '事实正确性',
  faithfulness: '忠实度',
  response_relevancy: '回答相关性',
  llm_context_recall: '上下文召回',
  id_based_context_precision: '上下文精确率(ID)',
  id_based_context_recall: '上下文召回率(ID)',
  retrieved_context_precision: '召回上下文精确率',
  prompt_context_precision: 'Prompt 上下文精确率',
  final_context_precision: '最终上下文精确率',
  'hit@1': 'HIT@1',
  'hit@3': 'HIT@3',
  'hit@5': 'HIT@5',
  'recall@3': 'Recall@3',
  'recall@5': 'Recall@5',
  'mrr@3': 'MRR@3',
  'mrr@5': 'MRR@5',
  avg_total_duration_ms: '平均总耗时',
  p50_total_duration_ms: 'P50 总耗时',
  p95_total_duration_ms: 'P95 总耗时',
  avg_first_token_ms: '平均首 token 延迟',
  p50_first_token_ms: 'P50 首 token 延迟',
  p95_first_token_ms: 'P95 首 token 延迟',
  avg_first_answer_ms: '平均首答耗时',
  p50_first_answer_ms: 'P50 首答耗时',
  p95_first_answer_ms: 'P95 首答耗时',
  cold_avg_total_duration_ms: '冷缓存平均总耗时',
  cold_p50_total_duration_ms: '冷缓存 P50 总耗时',
  cold_p95_total_duration_ms: '冷缓存 P95 总耗时',
  cold_avg_first_token_ms: '冷缓存平均首 token',
  cold_p50_first_token_ms: '冷缓存 P50 首 token',
  cold_p95_first_token_ms: '冷缓存 P95 首 token',
  cold_avg_first_answer_ms: '冷缓存平均首答',
  cold_p50_first_answer_ms: '冷缓存 P50 首答',
  cold_p95_first_answer_ms: '冷缓存 P95 首答',
  hot_avg_total_duration_ms: '热缓存平均总耗时',
  hot_p50_total_duration_ms: '热缓存 P50 总耗时',
  hot_p95_total_duration_ms: '热缓存 P95 总耗时',
  hot_avg_first_token_ms: '热缓存平均首 token',
  hot_p50_first_token_ms: '热缓存 P50 首 token',
  hot_p95_first_token_ms: '热缓存 P95 首 token',
  hot_avg_first_answer_ms: '热缓存平均首答',
  hot_p50_first_answer_ms: '热缓存 P50 首答',
  hot_p95_first_answer_ms: '热缓存 P95 首答',
  empty_retrieval_rate: '空检索率',
  empty_answer_rate: '空回答率',
  crag_retry_rate: 'CRAG 重试率',
  hallucination_retry_rate: '幻觉重试率',
  need_rag_rate: 'RAG 使用率',
  cache_hit_rate: '全链路缓存命中率',
  l0_cache_hit_rate: 'L0 命中率',
  l1_cache_hit_rate: 'L1 命中率',
  l2_cache_hit_rate: 'L2 命中率',
  retrieval_cache_rate: '检索缓存命中率',
  answer_cache_rate: '答案缓存命中率',
  avg_cache_writes: '平均缓存写入',
  llm_fallback_rate: '模型回退率',
  error_rate: '错误率',
  router_simple_rate: '简单问题占比',
  hyde_enabled_rate: 'HyDE 启用率',
  crag_router_enabled_rate: 'CRAG 启用率',
  anchor_enabled_rate: 'Anchor 启用率',
  avg_target_coverage_rate: '目标覆盖率',
  target_coverage_case_rate: '目标覆盖样本率',
  retrieval_judge_skipped_rate: '检索 Judge 跳过率',
  hallucination_judge_skipped_rate: '幻觉 Judge 跳过率',
  rerank_fallback_rate: 'Rerank 回退率',
  rerank_heuristic_rate: 'Rerank 启发式率',
  rerank_cache_hit_rate: 'Rerank 缓存命中率',
  avg_rerank_candidate_count: '平均重排候选数',
  avg_rerank_selected_count: '平均重排入选数',
  avg_final_context_docs: '平均最终上下文数',
  avg_final_context_used_chars: '平均上下文字数',
};

const CACHE_NAMESPACE_LABELS: Record<string, string> = {
  embedding: 'Embedding',
  retrieval_embedding: '向量检索',
  retrieval_bm25: 'BM25 检索',
  retrieval_kg: '图谱检索',
  hyde_doc: 'HyDE 文档',
  web_search: '联网搜索',
  rerank: '重排序',
  answer: '答案生成',
};

const GROUND_TRUTH_SOURCE_LABELS: Record<string, string> = {
  declared_ids: '沿用标注 chunk_id',
  reference_answer_bm25: '按参考答案重映射',
  unresolved: '未完成对齐',
};

const GROUND_TRUTH_REASON_LABELS: Record<string, string> = {
  no_retrieval_ground_truth: '未提供检索金标',
  item_name_not_in_corpus: 'item_name 不在当前知识库',
  no_candidate_chunks: '当前知识库没有候选切片',
  no_reference_match: '参考答案未命中候选切片',
  reference_match_too_weak: '参考答案与候选切片匹配过弱',
  declared_ids_stale: '历史 chunk_id 已失效',
  unknown: '未标注原因',
};

const HEADLINE_METRIC_KEYS = [
  'factual_correctness',
  'faithfulness',
  'response_relevancy',
  'recall@5',
  'mrr@5',
  'hit@5',
  'recall@3',
  'mrr@3',
  'hit@3',
  'retrieval_cache_rate',
  'avg_total_duration_ms',
  'p95_total_duration_ms',
  'p95_first_token_ms',
] as const;

const VARIANT_METRIC_KEYS = [
  'factual_correctness',
  'faithfulness',
  'response_relevancy',
  'recall@5',
  'mrr@5',
  'hit@5',
  'recall@3',
  'mrr@3',
  'hit@3',
  'retrieval_cache_rate',
  'avg_total_duration_ms',
  'p95_total_duration_ms',
  'p95_first_token_ms',
] as const;

const CACHE_METRIC_KEYS = [
  'cache_hit_rate',
  'l0_cache_hit_rate',
  'l1_cache_hit_rate',
  'l2_cache_hit_rate',
  'retrieval_cache_rate',
  'rerank_cache_hit_rate',
  'answer_cache_rate',
  'avg_cache_writes',
] as const;

const EVALUATION_METRIC_GROUPS = [
  {
    key: 'quality',
    label: '质量',
    metricKeys: [
      'factual_correctness',
      'faithfulness',
      'response_relevancy',
      'llm_context_recall',
      'id_based_context_precision',
      'id_based_context_recall',
      'final_context_precision',
    ],
  },
  {
    key: 'retrieval',
    label: '检索',
    metricKeys: [
      'recall@5',
      'mrr@5',
      'hit@5',
      'recall@3',
      'mrr@3',
      'hit@3',
      'retrieved_context_precision',
      'prompt_context_precision',
      'final_context_precision',
      'empty_retrieval_rate',
      'empty_answer_rate',
      'router_simple_rate',
      'hyde_enabled_rate',
      'anchor_enabled_rate',
      'crag_router_enabled_rate',
      'avg_target_coverage_rate',
      'target_coverage_case_rate',
      'avg_rerank_candidate_count',
      'avg_rerank_selected_count',
      'rerank_fallback_rate',
      'rerank_heuristic_rate',
      'retrieval_judge_skipped_rate',
      'hallucination_judge_skipped_rate',
      'avg_final_context_docs',
      'avg_final_context_used_chars',
    ],
  },
  {
    key: 'latency',
    label: '时延',
    metricKeys: [
      'avg_total_duration_ms',
      'p50_total_duration_ms',
      'p95_total_duration_ms',
      'avg_first_token_ms',
      'p50_first_token_ms',
      'p95_first_token_ms',
      'avg_first_answer_ms',
      'p50_first_answer_ms',
      'p95_first_answer_ms',
      'cold_avg_total_duration_ms',
      'cold_p50_total_duration_ms',
      'cold_p95_total_duration_ms',
      'hot_avg_total_duration_ms',
      'hot_p50_total_duration_ms',
      'hot_p95_total_duration_ms',
    ],
  },
  {
    key: 'cache',
    label: '缓存',
    metricKeys: CACHE_METRIC_KEYS,
  },
] as const;

type EvaluationMetricGroupKey = (typeof EVALUATION_METRIC_GROUPS)[number]['key'];

const FRONTEND_EVALUATION_TEMPLATE_PATH = '/app/docs/graph_eval_cases.docs.json';
const HIDDEN_EVALUATION_VARIANTS = new Set(['neo4j_graph_first']);
const VARIANT_CATEGORY_ORDER = ['base', 'agentic', 'router'];
const VARIANT_CATEGORY_LABELS: Record<string, string> = {
  base: '纯 Base',
  agentic: 'Agentic 增强',
  router: 'Router 版',
};
const VARIANT_CATEGORY_DESCRIPTIONS: Record<string, string> = {
  base: '基础检索与传统组合对照，用来判断每个基础组件的单独收益。',
  agentic: '在基础检索上加入上下文扩展、检索补救、结构化回答或缓存。',
  router: '由 Router 控制 HyDE、CRAG、Anchor 等路径，验证质量与耗时权衡。',
};
const FEATURE_CATEGORY_ORDER = ['retrieval', 'agentic', 'quality', 'performance', 'external'];
const FEATURE_CATEGORY_LABELS: Record<string, string> = {
  retrieval: '检索增强',
  agentic: 'Agentic',
  quality: '质量控制',
  performance: '性能缓存',
  external: '外部功能',
};
const MAX_FEATURE_COMPARISONS = 6;
const BASELINE_COMBO_ID = 'baseline';

interface FeatureComboItem {
  id: string;
  label: string;
  features: string[];
  locked?: boolean;
}

const BASELINE_COMBO: FeatureComboItem = {
  id: BASELINE_COMBO_ID,
  label: 'Baseline',
  features: [],
  locked: true,
};

function isVisibleEvaluationVariant(variantName?: string | null) {
  return Boolean(variantName) && !HIDDEN_EVALUATION_VARIANTS.has(String(variantName));
}

function inferVariantCategory(variant: EvaluationVariantOption) {
  const name = variant.name || '';
  if (name.startsWith('router_')) return 'router';
  if (name.startsWith('agentic_') || name === 'final_system') return 'agentic';
  return 'base';
}

function featureOrderMap(features: EvaluationFeatureOption[]) {
  return new Map(features.map((feature, index) => [feature.key, index]));
}

function normalizeFeatureKeys(keys: string[], features: EvaluationFeatureOption[]) {
  const order = featureOrderMap(features);
  return Array.from(new Set(keys.filter((key) => order.has(key)))).sort(
    (left, right) => (order.get(left) ?? 999) - (order.get(right) ?? 999),
  );
}

function resolveFeatureKeys(keys: string[], features: EvaluationFeatureOption[]) {
  const featureMap = new Map(features.map((feature) => [feature.key, feature]));
  const resolved = new Set(normalizeFeatureKeys(keys, features));
  let changed = true;
  while (changed) {
    changed = false;
    Array.from(resolved).forEach((key) => {
      (featureMap.get(key)?.dependencies || []).forEach((dependency) => {
        if (featureMap.has(dependency) && !resolved.has(dependency)) {
          resolved.add(dependency);
          changed = true;
        }
      });
    });
  }
  return normalizeFeatureKeys(Array.from(resolved), features);
}

function featureSignature(keys: string[], features: EvaluationFeatureOption[]) {
  return resolveFeatureKeys(keys, features).join('|');
}

function variantFeatureSignature(variant?: { feature_variant?: { resolved_features?: string[]; requested_features?: string[] } } | null) {
  const featureVariant = variant?.feature_variant;
  const keys = Array.isArray(featureVariant?.resolved_features)
    ? featureVariant?.resolved_features
    : featureVariant?.requested_features;
  if (!Array.isArray(keys)) return null;
  return keys.map((key) => String(key).trim()).filter(Boolean).join('|');
}

function featureLabelMap(features: EvaluationFeatureOption[]) {
  return new Map(features.map((feature) => [feature.key, feature.label]));
}

function formatFeatureComboLabel(keys: string[], features: EvaluationFeatureOption[]) {
  const labels = resolveFeatureKeys(keys, features)
    .map((key) => featureLabelMap(features).get(key) || key)
    .filter(Boolean);
  return labels.length ? `Baseline + ${labels.join(' + ')}` : 'Baseline';
}

function featureComboToSpec(combo: FeatureComboItem, features: EvaluationFeatureOption[]): EvaluationFeatureVariantSpec {
  return {
    label: combo.label || formatFeatureComboLabel(combo.features, features),
    features: normalizeFeatureKeys(combo.features, features),
  };
}

function upsertJob(list: EvaluationJob[], nextJob: EvaluationJob) {
  const existing = list.find((job) => job.job_id === nextJob.job_id);
  if (!existing) return [nextJob, ...list];
  return list.map((job) => (job.job_id === nextJob.job_id ? nextJob : job));
}

function formatStageLabel(stage: string) {
  return STAGE_LABELS[stage] || stage.replaceAll('_', ' ');
}

function formatMetricLabel(metricKey: string) {
  return METRIC_LABELS[metricKey] || metricKey.replaceAll('_', ' ');
}

function formatQueryTypeLabel(queryType: string) {
  return QUERY_TYPE_LABELS[queryType] || queryType;
}

function formatNamespaceLabel(namespace: string) {
  return CACHE_NAMESPACE_LABELS[namespace] || namespace;
}

function formatVariantLabel(variantName: string, technique?: string) {
  if (technique) return technique;
  return variantName
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

function formatRouterDecision(value?: string | null) {
  const mapping: Record<string, string> = {
    default_path: '默认路径',
    simple_fast_path: '简单快路径',
    complex_deep_path: '复杂深路径',
    anchor_grounded_path: 'Anchor 证据路径',
  };
  if (!value) return '-';
  return mapping[value] || value;
}

function formatReportTimestamp(value?: string | null) {
  if (!value) return '未知时间';
  return value.replace('T', ' ').slice(0, 19);
}

function formatMs(value: number | null | undefined) {
  if (value == null) return '-';
  if (value >= 1000) return `${(value / 1000).toFixed(2)}s`;
  return `${Math.round(value)}ms`;
}

function toNumericValue(value: unknown) {
  if (typeof value === 'number' && Number.isFinite(value)) return value;
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number(value);
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
}

function isDurationMetric(metricKey: string) {
  return metricKey.includes('duration_ms') || metricKey.endsWith('_ms');
}

function isRatioMetric(metricKey: string) {
  return (
    metricKey.includes('rate') ||
    metricKey.startsWith('hit@') ||
    metricKey.startsWith('recall@') ||
    metricKey.startsWith('mrr@') ||
    [
      'factual_correctness',
      'faithfulness',
      'response_relevancy',
      'llm_context_recall',
      'id_based_context_precision',
      'id_based_context_recall',
      'retrieved_context_precision',
      'prompt_context_precision',
      'final_context_precision',
    ].includes(metricKey)
  );
}

function getMetricValueClass(metricKey: string, value: number | null | undefined) {
  if (
    ['rerank_fallback_rate', 'rerank_heuristic_rate', 'error_rate', 'empty_retrieval_rate'].includes(metricKey)
    && typeof value === 'number'
    && value > 0
  ) {
    return 'text-red-600 dark:text-red-300';
  }
  return 'text-gray-900 dark:text-white';
}

function formatNumber(value: number | null | undefined, digits = 3) {
  if (value == null) return '-';
  return Number(value).toFixed(digits).replace(/\.?0+$/, '');
}

function formatMetricValue(metricKey: string, value: number | null | undefined) {
  if (value == null) return '-';
  if (isDurationMetric(metricKey)) return formatMs(value);
  if (isRatioMetric(metricKey)) return `${(value * 100).toFixed(1)}%`;
  return formatNumber(value);
}

function formatDelta(metricKey: string, delta: EvaluationMetricDelta | undefined) {
  if (!delta || delta.delta == null) return '-';
  const sign = delta.delta > 0 ? '+' : delta.delta < 0 ? '-' : '';
  const absValue = Math.abs(delta.delta);
  if (isDurationMetric(metricKey)) return `${sign}${formatMs(absValue)}`;
  if (isRatioMetric(metricKey)) return `${sign}${(absValue * 100).toFixed(1)}pp`;
  return `${sign}${formatNumber(absValue)}`;
}

function formatJobStatus(status: EvaluationJob['status']) {
  const mapping: Record<EvaluationJob['status'], string> = {
    pending: '等待中',
    running: '运行中',
    cancelling: '停止中',
    cancelled: '已取消',
    completed: '已完成',
    failed: '失败',
  };
  return mapping[status] || status;
}

function formatEvaluationPhase(phase?: EvaluationJob['phase']) {
  const mapping: Record<NonNullable<EvaluationJob['phase']>, string> = {
    pending: '等待中',
    loading_dataset: '加载数据集',
    dataset_loaded: '数据集已加载',
    variant_started: '初始化变体',
    warmup: '缓存预热',
    evaluation: '正式评测',
    variant_completed: '变体完成',
    saving_report: '写入报告',
    report_saved: '报告已生成',
    cancelling: '停止中',
    cancelled: '已取消',
    completed: '已完成',
    failed: '失败',
  };
  return phase ? mapping[phase] || phase : '-';
}

function resolveTemplateDatasetPath(rawPath?: string | null) {
  const value = rawPath?.trim();
  if (!value) return FRONTEND_EVALUATION_TEMPLATE_PATH;
  const normalized = value.replaceAll('\\', '/');
  if (normalized.endsWith('graph_eval_cases.docs.json')) return normalized;
  if (normalized.endsWith('/test/graph_eval_cases.template.json')) {
    return normalized.replace('/test/graph_eval_cases.template.json', '/docs/graph_eval_cases.docs.json');
  }
  if (normalized.endsWith('graph_eval_cases.template.json')) {
    return normalized.replace('graph_eval_cases.template.json', 'graph_eval_cases.docs.json');
  }
  return normalized;
}

function getSummaryMetric(summary: EvaluationSummary | null | undefined, metricKey: string) {
  if (!summary) return null;
  if (metricKey in summary.headline_metrics) return summary.headline_metrics[metricKey] ?? null;
  if (metricKey in summary.ragas_metrics) return summary.ragas_metrics[metricKey] ?? null;
  if (metricKey in summary.retrieval_metrics) return summary.retrieval_metrics[metricKey] ?? null;
  if (metricKey in summary.pipeline_metrics) return summary.pipeline_metrics[metricKey] ?? null;
  const perfMetric = summary.performance_metrics?.[metricKey];
  return typeof perfMetric === 'number' ? perfMetric : null;
}

function getMetricCoverage(summary: EvaluationSummary | null | undefined, metricKey: string) {
  if (!summary) return null;
  if (summary.ragas_coverage?.[metricKey] != null) return summary.ragas_coverage[metricKey];
  if (summary.retrieval_coverage?.[metricKey] != null) return summary.retrieval_coverage[metricKey];
  return null;
}

function formatCoverageLabel(summary: EvaluationSummary | null | undefined, metricKey: string) {
  const coverage = getMetricCoverage(summary, metricKey);
  return coverage == null ? undefined : `覆盖样本 ${coverage}`;
}

function formatBreakdown(
  breakdown: Record<string, number> | null | undefined,
  labelMap: Record<string, string> = {},
) {
  const entries = Object.entries(breakdown || {});
  if (!entries.length) return '-';
  return entries
    .map(([key, value]) => `${labelMap[key] || key}: ${value}`)
    .join(' · ');
}

function formatGroundTruthSourceBreakdown(summary: EvaluationRetrievalGroundTruthSummary | null | undefined) {
  const labels = { ...GROUND_TRUTH_SOURCE_LABELS };
  if (summary && summary.unresolved_cases === 0 && (summary.source_breakdown?.unresolved || 0) > 0) {
    labels.unresolved = '原始 item 未命中已兜底';
  }
  return formatBreakdown(summary?.source_breakdown, labels);
}

function formatQualityJudgeSummary(metadata: Record<string, unknown> | null | undefined) {
  if (!metadata) return '';
  const qualityMode = typeof metadata.quality_mode === 'string' ? metadata.quality_mode : '';
  const promptVersion = typeof metadata.judge_prompt_version === 'string' ? metadata.judge_prompt_version : '';
  const llmErrorCount = typeof metadata.llm_error_count === 'number' ? metadata.llm_error_count : null;
  const methodCounts =
    metadata.method_counts && typeof metadata.method_counts === 'object'
      ? (metadata.method_counts as Record<string, unknown>)
      : null;
  const methodText = methodCounts
    ? Object.entries(methodCounts)
      .map(([key, value]) => `${key}: ${value}`)
      .join(' · ')
    : '';
  return [
    qualityMode ? `评估模式 ${qualityMode}` : '',
    promptVersion ? `Judge ${promptVersion}` : '',
    methodText,
    llmErrorCount && llmErrorCount > 0 ? `Judge 失败 ${llmErrorCount} 次` : '',
  ]
    .filter(Boolean)
    .join(' · ');
}

function formatReportVariantSummary(
  report: EvaluationReportListItem,
  techniqueMap: Record<string, string>,
) {
  const visibleVariants = report.variants.filter((variantName) => isVisibleEvaluationVariant(variantName));
  const labels = visibleVariants
    .map((variantName) => formatVariantLabel(variantName, techniqueMap[variantName]))
    .filter(Boolean);
  if (labels.length === 0) return report.dataset_name || report.file_name;
  if (labels.length === 1) return labels[0];
  if (labels.length === 2) return `${labels[0]} vs ${labels[1]}`;
  const finalVariantName = isVisibleEvaluationVariant(report.final_variant)
    ? report.final_variant
    : visibleVariants[visibleVariants.length - 1] || visibleVariants[0] || report.final_variant;
  const finalLabel = formatVariantLabel(finalVariantName, techniqueMap[finalVariantName]);
  return `${finalLabel} · ${labels.length} 个方案`;
}

function formatCompactVariantLabel(
  variantName?: string | null,
  variant?: { technique?: string; feature_variant?: { feature_labels?: string[] } } | null,
) {
  if (!variantName) return '-';
  const featureCount = variant?.feature_variant?.feature_labels?.length || 0;
  if (featureCount > 0) return `Baseline + ${featureCount} 个功能`;
  const label = formatVariantLabel(variantName, variant?.technique);
  return label.length > 28 ? `${label.slice(0, 26)}...` : label;
}

function StatCard({
  title,
  value,
  subtitle,
  icon,
  loading = false,
}: {
  title: string;
  value: string;
  subtitle?: string;
  icon: React.ReactNode;
  loading?: boolean;
}) {
  if (loading) {
    return (
      <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900/60 p-4 shadow-sm">
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0 flex-1">
            <p className="text-xs uppercase tracking-[0.15em] text-gray-500">{title}</p>
            <div className="mt-3 h-8 w-28 animate-pulse rounded bg-gray-200 dark:bg-gray-800" />
            <div className="mt-2 h-3 w-36 animate-pulse rounded bg-gray-100 dark:bg-gray-800/70" />
          </div>
          <div className="rounded-xl bg-violet-100 p-2 text-violet-600 dark:bg-violet-500/10 dark:text-violet-400">
            {icon}
          </div>
        </div>
      </div>
    );
  }
  return (
    <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900/60 p-4 shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0 flex-1">
          <p className="text-xs uppercase tracking-[0.15em] text-gray-500">{title}</p>
          <p className="mt-2 truncate text-2xl font-bold text-gray-900 dark:text-white" title={value}>
            {value}
          </p>
          {subtitle && (
            <p className="mt-1 line-clamp-2 text-xs text-gray-500" title={subtitle}>
              {subtitle}
            </p>
          )}
        </div>
        <div className="rounded-xl bg-violet-100 p-2 text-violet-600 dark:bg-violet-500/10 dark:text-violet-400">
          {icon}
        </div>
      </div>
    </div>
  );
}

function LoadingSection({ title, icon }: { title: string; icon: React.ReactNode }) {
  return (
    <section className="rounded-2xl border border-gray-200 bg-white p-4 shadow-sm dark:border-gray-800 dark:bg-gray-900/60">
      <div className="mb-4 flex items-center gap-2">
        {icon}
        <h2 className="text-lg font-semibold">{title}</h2>
      </div>
      <div className="flex h-[220px] items-center justify-center text-sm text-gray-500">
        <Loader2 className="mr-2 h-5 w-5 animate-spin text-violet-500" />
        正在加载图表数据...
      </div>
    </section>
  );
}

function SectionMessage({
  title,
  message,
  icon,
}: {
  title: string;
  message: string;
  icon: React.ReactNode;
}) {
  return (
    <section className="rounded-2xl border border-dashed border-gray-300 bg-white p-6 shadow-sm dark:border-gray-700 dark:bg-gray-900/60">
      <div className="flex flex-col items-center justify-center gap-3 text-center">
        <div className="rounded-full bg-violet-100 p-3 text-violet-600 dark:bg-violet-500/10 dark:text-violet-400">
          {icon}
        </div>
        <div>
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">{title}</h2>
          <p className="mt-1 text-sm text-gray-500">{message}</p>
        </div>
      </div>
    </section>
  );
}

export default function PerformancePage() {
  const [activeTab, setActiveTab] = useState<'runtime' | 'evaluation'>('runtime');
  const [granularity, setGranularity] = useState<'day' | 'hour'>('day');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');
  const [summary, setSummary] = useState<PerformanceSummary | null>(null);
  const [timeSeries, setTimeSeries] = useState<PerformanceTimePoint[]>([]);
  const [stages, setStages] = useState<StageBreakdown[]>([]);
  const [performanceLoading, setPerformanceLoading] = useState(true);
  const [performanceError, setPerformanceError] = useState<string | null>(null);
  const [evaluationReports, setEvaluationReports] = useState<EvaluationReportListItem[]>([]);
  const [selectedReportId, setSelectedReportId] = useState('');
  const [evaluationMeta, setEvaluationMeta] = useState<EvaluationReportListItem | null>(null);
  const [evaluationReport, setEvaluationReport] = useState<EvaluationReportDetail | null>(null);
  const [evaluationLoading, setEvaluationLoading] = useState(true);
  const [evaluationError, setEvaluationError] = useState<string | null>(null);
  const [evaluationConfig, setEvaluationConfig] = useState<EvaluationConfig | null>(null);
  const [evaluationJobs, setEvaluationJobs] = useState<EvaluationJob[]>([]);
  const [activeEvaluationJob, setActiveEvaluationJob] = useState<EvaluationJob | null>(null);
  const [datasetPath, setDatasetPath] = useState('');
  const [includeBaseline, setIncludeBaseline] = useState(false);
  const [featureCombos, setFeatureCombos] = useState<FeatureComboItem[]>([]);
  const [editingComboId, setEditingComboId] = useState<string | null>(null);
  const [draftFeatureKeys, setDraftFeatureKeys] = useState<string[]>([]);
  const [draftComboLabel, setDraftComboLabel] = useState('');
  const [appendEditingComboId, setAppendEditingComboId] = useState<string | null>(null);
  const [appendFeatureKeys, setAppendFeatureKeys] = useState<string[]>([]);
  const [appendComboLabel, setAppendComboLabel] = useState('');
  const [appendingEvaluation, setAppendingEvaluation] = useState(false);
  const [activeMetricGroupKey, setActiveMetricGroupKey] = useState<EvaluationMetricGroupKey>('quality');
  const [evaluationSubTab, setEvaluationSubTab] = useState<'run' | 'report' | 'trial' | 'cache'>('run');
  const [selectedVariantName, setSelectedVariantName] = useState('');
  const [trialFeatureKeys, setTrialFeatureKeys] = useState<string[]>(['hyde', 'bm25']);
  const [trialComboLabel, setTrialComboLabel] = useState('');
  const [trialQuery, setTrialQuery] = useState('');
  const [trialResult, setTrialResult] = useState<EvaluationVariantTrialResult | null>(null);
  const [trialLoading, setTrialLoading] = useState(false);
  const [trialError, setTrialError] = useState<string | null>(null);
  const [startingEvaluation, setStartingEvaluation] = useState(false);
  const [evaluationActionError, setEvaluationActionError] = useState<string | null>(null);
  const [evaluationActionMessage, setEvaluationActionMessage] = useState<string | null>(null);
  const [syncingEvaluationDataset, setSyncingEvaluationDataset] = useState(false);
  const [datasetSyncResult, setDatasetSyncResult] = useState<EvaluationDatasetSyncResult | null>(null);
  const [migratingKnowledgeBase, setMigratingKnowledgeBase] = useState(false);
  const [chunkMigrationResult, setChunkMigrationResult] = useState<ChunkIdMigrationResult | null>(null);
  const [queryCacheStats, setQueryCacheStats] = useState<QueryCacheStats | null>(null);
  const [queryCacheLoading, setQueryCacheLoading] = useState(false);
  const [queryCacheError, setQueryCacheError] = useState<string | null>(null);
  const [queryCacheActionMessage, setQueryCacheActionMessage] = useState<string | null>(null);
  const [resettingQueryCache, setResettingQueryCache] = useState(false);

  const loadPerformanceData = useCallback(async () => {
    setPerformanceLoading(true);
    setPerformanceError(null);
    try {
      const [summaryData, timeSeriesData, stageData] = await Promise.all([
        getPerformanceSummaryData(startDate || undefined, endDate || undefined),
        getPerformanceTimeSeries(granularity, startDate || undefined, endDate || undefined),
        getStageBreakdown(startDate || undefined, endDate || undefined),
      ]);
      setSummary(summaryData);
      setTimeSeries(timeSeriesData);
      setStages(stageData);
    } catch (err: unknown) {
      setPerformanceError(err instanceof Error && err.message ? err.message : '加载性能数据失败');
    } finally {
      setPerformanceLoading(false);
    }
  }, [granularity, startDate, endDate]);

  const loadEvaluationData = useCallback(async (preferredReportId?: string) => {
    setEvaluationLoading(true);
    setEvaluationError(null);
    setDatasetSyncResult(null);
    setChunkMigrationResult(null);
    try {
      const reportsData = await getEvaluationReports();
      const reports = reportsData.reports || [];
      setEvaluationReports(reports);

      const nextReportId =
        preferredReportId && reports.some((report) => report.report_id === preferredReportId)
          ? preferredReportId
          : reportsData.latest_report_id || reports[0]?.report_id || '';

      setSelectedReportId(nextReportId);

      if (!nextReportId) {
        setEvaluationMeta(null);
        setEvaluationReport(null);
        return;
      }

      const detailData = await getEvaluationReport(nextReportId);
      setEvaluationMeta(detailData.meta);
      setEvaluationReport(detailData.report);
    } catch (err: unknown) {
      setEvaluationError(err instanceof Error && err.message ? err.message : '加载量化评测失败');
    } finally {
      setEvaluationLoading(false);
    }
  }, []);

  const loadEvaluationRuntime = useCallback(async () => {
    try {
      const [configData, jobsData] = await Promise.all([getEvaluationConfig(), getEvaluationJobs()]);
      setEvaluationConfig(configData);
      setDatasetPath((prev) => prev || resolveTemplateDatasetPath(configData.template_dataset_path));
      const jobs = jobsData.jobs || [];
      setEvaluationJobs(jobs);
      setActiveEvaluationJob((prev) => {
        if (prev && ['pending', 'running', 'cancelling'].includes(prev.status)) {
          const updated = jobs.find((job) => job.job_id === prev.job_id);
          return updated || prev;
        }
        return jobs[0] || null;
      });
    } catch (err: unknown) {
      setEvaluationActionError(err instanceof Error && err.message ? err.message : '加载评测配置失败');
    }
  }, []);

  const loadQueryCacheRuntime = useCallback(async () => {
    setQueryCacheLoading(true);
    setQueryCacheError(null);
    try {
      const stats = await getQueryCacheStats();
      setQueryCacheStats(stats);
    } catch (err: unknown) {
      setQueryCacheError(err instanceof Error && err.message ? err.message : '加载缓存统计失败');
    } finally {
      setQueryCacheLoading(false);
    }
  }, []);

  const handleReportChange = useCallback(async (reportId: string) => {
    setSelectedReportId(reportId);
    setEvaluationLoading(true);
    setEvaluationError(null);
    try {
      const detailData = await getEvaluationReport(reportId);
      setEvaluationMeta(detailData.meta);
      setEvaluationReport(detailData.report);
    } catch (err: unknown) {
      setEvaluationError(err instanceof Error && err.message ? err.message : '加载量化评测失败');
    } finally {
      setEvaluationLoading(false);
    }
  }, []);

  const evaluationFeatures = evaluationConfig?.feature_catalog || [];
  const runComparisonCount = featureCombos.length + (includeBaseline ? 1 : 0);

  const handleToggleDraftFeature = useCallback((featureKey: string) => {
    setDraftFeatureKeys((prev) =>
      prev.includes(featureKey)
        ? prev.filter((item) => item !== featureKey)
        : normalizeFeatureKeys([...prev, featureKey], evaluationFeatures),
    );
  }, [evaluationFeatures]);

  const handleToggleTrialFeature = useCallback((featureKey: string) => {
    setTrialFeatureKeys((prev) =>
      prev.includes(featureKey)
        ? prev.filter((item) => item !== featureKey)
        : normalizeFeatureKeys([...prev, featureKey], evaluationFeatures),
    );
  }, [evaluationFeatures]);

  const handleToggleAppendFeature = useCallback((featureKey: string) => {
    setAppendFeatureKeys((prev) =>
      prev.includes(featureKey)
        ? prev.filter((item) => item !== featureKey)
        : normalizeFeatureKeys([...prev, featureKey], evaluationFeatures),
    );
  }, [evaluationFeatures]);

  const handleStartAddCombo = useCallback(() => {
    setEditingComboId('new');
    setDraftFeatureKeys([]);
    setDraftComboLabel('');
    setEvaluationActionError(null);
  }, []);

  const handleStartAppendCombo = useCallback(() => {
    setAppendEditingComboId('new');
    setAppendFeatureKeys([]);
    setAppendComboLabel('');
    setEvaluationActionError(null);
  }, []);

  const handleEditCombo = useCallback((combo: FeatureComboItem) => {
    if (combo.locked) return;
    setEditingComboId(combo.id);
    setDraftFeatureKeys(combo.features);
    setDraftComboLabel(combo.label);
    setEvaluationActionError(null);
  }, []);

  const handleSaveCombo = useCallback(() => {
    const resolvedSignature = featureSignature(draftFeatureKeys, evaluationFeatures);
    if (!resolvedSignature) {
      setEvaluationActionError('请至少选择一个增强功能；Baseline 可在对比项列表中单独勾选。');
      return;
    }
    const duplicate = featureCombos.some(
      (combo) => combo.id !== editingComboId && featureSignature(combo.features, evaluationFeatures) === resolvedSignature,
    );
    if (duplicate) {
      setEvaluationActionError('这个功能组合已经存在，请调整后再保存。');
      return;
    }
    const isNew = editingComboId === 'new' || !editingComboId;
    if (isNew && runComparisonCount >= MAX_FEATURE_COMPARISONS) {
      setEvaluationActionError(`最多选择 ${MAX_FEATURE_COMPARISONS} 个对比项。`);
      return;
    }
    const nextFeatures = normalizeFeatureKeys(draftFeatureKeys, evaluationFeatures);
    const nextLabel = draftComboLabel.trim() || formatFeatureComboLabel(nextFeatures, evaluationFeatures);
    const nextCombo: FeatureComboItem = {
      id: isNew ? `combo-${Date.now()}` : editingComboId,
      label: nextLabel,
      features: nextFeatures,
    };
    setFeatureCombos((prev) =>
      isNew ? [...prev, nextCombo] : prev.map((combo) => (combo.id === editingComboId ? nextCombo : combo)),
    );
    setEditingComboId(null);
    setDraftFeatureKeys([]);
    setDraftComboLabel('');
    setEvaluationActionError(null);
  }, [draftComboLabel, draftFeatureKeys, editingComboId, evaluationFeatures, featureCombos, runComparisonCount]);

  const handleDeleteCombo = useCallback((comboId: string) => {
    setFeatureCombos((prev) => prev.filter((combo) => combo.id !== comboId));
    if (editingComboId === comboId) {
      setEditingComboId(null);
      setDraftFeatureKeys([]);
      setDraftComboLabel('');
    }
  }, [editingComboId]);

  const handleStartEvaluation = useCallback(async () => {
    const trimmedDatasetPath = datasetPath.trim();
    if (!trimmedDatasetPath) {
      setEvaluationActionError('请先填写评测数据集路径');
      return;
    }
    const selectedCombos = [
      ...(includeBaseline ? [BASELINE_COMBO] : []),
      ...featureCombos,
    ];
    if (selectedCombos.length === 0) {
      setEvaluationActionError('请至少选择一个对比项');
      return;
    }

    setStartingEvaluation(true);
    setEvaluationActionError(null);
    setEvaluationActionMessage(null);
    setDatasetSyncResult(null);
    setChunkMigrationResult(null);
    try {
      const job = await createEvaluationJob({
        dataset_path: trimmedDatasetPath,
        variants: [],
        feature_variants: selectedCombos.map((combo) => featureComboToSpec(combo, evaluationFeatures)),
      });
      setActiveEvaluationJob(job);
      setEvaluationJobs((prev) => upsertJob(prev, job));
      void loadEvaluationRuntime();
    } catch (err: unknown) {
      setEvaluationActionError(err instanceof Error && err.message ? err.message : '启动评测失败');
    } finally {
      setStartingEvaluation(false);
    }
  }, [datasetPath, evaluationFeatures, featureCombos, includeBaseline, loadEvaluationRuntime]);

  const handleAppendEvaluation = useCallback(async () => {
    if (!selectedReportId || !evaluationReport) {
      setEvaluationActionError('请先选择一个已完成的评测报告');
      return;
    }
    const resolvedFeatures = normalizeFeatureKeys(appendFeatureKeys, evaluationFeatures);
    const resolvedSignature = featureSignature(resolvedFeatures, evaluationFeatures);
    if (!resolvedSignature) {
      setEvaluationActionError('请至少选择一个要追加的增强功能');
      return;
    }
    const existingSignatures = new Set(
      Object.values(evaluationReport.variants || {})
        .map((variant) => variantFeatureSignature(variant))
        .filter((signature): signature is string => signature != null),
    );
    if (existingSignatures.has(resolvedSignature)) {
      setEvaluationActionError('这个功能组合已存在于当前报告，无法重复追加。');
      return;
    }

    setAppendingEvaluation(true);
    setEvaluationActionError(null);
    setEvaluationActionMessage(null);
    try {
      const job = await appendEvaluationReport(selectedReportId, {
        feature_variants: [
          {
            label: appendComboLabel.trim() || formatFeatureComboLabel(resolvedFeatures, evaluationFeatures),
            features: resolvedFeatures,
          },
        ],
      });
      setActiveEvaluationJob(job);
      setEvaluationJobs((prev) => upsertJob(prev, job));
      setAppendEditingComboId(null);
      setAppendFeatureKeys([]);
      setAppendComboLabel('');
      void loadEvaluationRuntime();
    } catch (err: unknown) {
      setEvaluationActionError(err instanceof Error && err.message ? err.message : '追加测评失败');
    } finally {
      setAppendingEvaluation(false);
    }
  }, [
    appendComboLabel,
    appendFeatureKeys,
    evaluationFeatures,
    evaluationReport,
    loadEvaluationRuntime,
    selectedReportId,
  ]);

  const handleCancelEvaluation = useCallback(async () => {
    if (!activeEvaluationJob) return;
    if (!['pending', 'running', 'cancelling'].includes(activeEvaluationJob.status)) return;

    setEvaluationActionError(null);
    setEvaluationActionMessage(null);
    try {
      const job = await cancelEvaluationJob(activeEvaluationJob.job_id);
      setActiveEvaluationJob(job);
      setEvaluationJobs((prev) => upsertJob(prev, job));
      setEvaluationActionMessage(
        job.status === 'cancelled' ? '评测任务已取消。' : '已向后台发送停止评测请求。',
      );
      void loadEvaluationRuntime();
    } catch (err: unknown) {
      setEvaluationActionError(err instanceof Error && err.message ? err.message : '停止评测失败');
    }
  }, [activeEvaluationJob, loadEvaluationRuntime]);

  const handleRunVariantTrial = useCallback(async () => {
    const trimmedQuery = trialQuery.trim();
    if (!trimmedQuery) {
      setTrialError('请先输入要验证的问题');
      return;
    }
    if (!trialFeatureKeys.length) {
      setTrialError('请至少选择一个试跑功能');
      return;
    }

    setTrialLoading(true);
    setTrialError(null);
    try {
      const result = await testEvaluationVariant({
        query: trimmedQuery,
        variant_spec: {
          label: trialComboLabel.trim() || formatFeatureComboLabel(trialFeatureKeys, evaluationFeatures),
          features: normalizeFeatureKeys(trialFeatureKeys, evaluationFeatures),
        },
      });
      setTrialResult(result);
    } catch (err: unknown) {
      setTrialError(err instanceof Error && err.message ? err.message : '在线试跑失败');
    } finally {
      setTrialLoading(false);
    }
  }, [evaluationFeatures, trialComboLabel, trialFeatureKeys, trialQuery]);

  const handleDeleteSelectedReport = useCallback(async () => {
    if (!selectedReportId) return;
    const reportLabel = evaluationMeta?.file_name || selectedReportId;
    if (!window.confirm(`确认删除评测报告 ${reportLabel} 吗？此操作不可撤销。`)) {
      return;
    }

    setEvaluationActionError(null);
    setEvaluationActionMessage(null);
    try {
      await deleteEvaluationReport(selectedReportId);
      setEvaluationActionMessage(`已删除评测报告：${reportLabel}`);
      await loadEvaluationData();
      await loadEvaluationRuntime();
    } catch (err: unknown) {
      setEvaluationActionError(err instanceof Error && err.message ? err.message : '删除评测报告失败');
    }
  }, [evaluationMeta?.file_name, loadEvaluationData, loadEvaluationRuntime, selectedReportId]);

  const handleSyncEvaluationDataset = useCallback(async () => {
    const targetDatasetPath = (
      evaluationMeta?.dataset_path ||
      evaluationReport?.dataset_path ||
      datasetPath
    ).trim();
    if (!targetDatasetPath) {
      setEvaluationActionError('当前没有可同步的评测数据集路径');
      return;
    }

    setSyncingEvaluationDataset(true);
    setEvaluationActionError(null);
    setEvaluationActionMessage(null);
    try {
      const result = await syncEvaluationDatasetChunkIds({
        dataset_path: targetDatasetPath,
        create_backup: true,
      });
      setDatasetSyncResult(result);
      setDatasetPath(result.output_path || targetDatasetPath);
      setEvaluationActionMessage(
        result.backup_path
          ? `${result.message} 已生成备份：${result.backup_path}`
          : result.message,
      );
    } catch (err: unknown) {
      setEvaluationActionError(err instanceof Error && err.message ? err.message : '同步评测集失败');
    } finally {
      setSyncingEvaluationDataset(false);
    }
  }, [datasetPath, evaluationMeta?.dataset_path, evaluationReport?.dataset_path]);

  const handleMigrateKnowledgeBaseChunkIds = useCallback(async () => {
    setMigratingKnowledgeBase(true);
    setEvaluationActionError(null);
    setEvaluationActionMessage(null);
    try {
      const migration = await migrateKnowledgeBaseChunkIds({
        dry_run: false,
        sync_graph: true,
      });
      setChunkMigrationResult(migration);

      let message = migration.message;
      const targetDatasetPath = (
        evaluationMeta?.dataset_path ||
        evaluationReport?.dataset_path ||
        datasetPath
      ).trim();
      const canAutoSyncDataset = Boolean(targetDatasetPath) && !targetDatasetPath.startsWith('/app/docs/');

      if (canAutoSyncDataset) {
        try {
          const syncResult = await syncEvaluationDatasetChunkIds({
            dataset_path: targetDatasetPath,
            create_backup: true,
          });
          setDatasetSyncResult(syncResult);
          setDatasetPath(syncResult.output_path || targetDatasetPath);
          message = `${migration.message} ${syncResult.message}`;
        } catch (syncErr: unknown) {
          const syncMessage =
            syncErr instanceof Error && syncErr.message ? syncErr.message : '评测集同步失败';
          message = `${migration.message} 评测集未自动同步：${syncMessage}`;
        }
      } else if (targetDatasetPath) {
        message = `${migration.message} 默认评测集已在工作区更新，重跑评测即可使用稳定 chunk_id。`;
      }

      setEvaluationActionMessage(message);
    } catch (err: unknown) {
      setEvaluationActionError(err instanceof Error && err.message ? err.message : '迁移知识库稳定 ID 失败');
    } finally {
      setMigratingKnowledgeBase(false);
    }
  }, [datasetPath, evaluationMeta?.dataset_path, evaluationReport?.dataset_path]);

  const handleResetQueryCache = useCallback(async () => {
    setResettingQueryCache(true);
    setQueryCacheError(null);
    setQueryCacheActionMessage(null);
    try {
      const result = await resetQueryCache('frontend_quant_eval_manual_reset');
      setQueryCacheActionMessage(result.message);
      await loadQueryCacheRuntime();
    } catch (err: unknown) {
      setQueryCacheError(err instanceof Error && err.message ? err.message : '清空查询缓存失败');
    } finally {
      setResettingQueryCache(false);
    }
  }, [loadQueryCacheRuntime]);

  const handleRefresh = useCallback(() => {
    if (activeTab === 'runtime') {
      void loadPerformanceData();
      return;
    }
    void loadEvaluationData(selectedReportId || undefined);
    void loadEvaluationRuntime();
    void loadQueryCacheRuntime();
  }, [activeTab, loadEvaluationData, loadEvaluationRuntime, loadPerformanceData, loadQueryCacheRuntime, selectedReportId]);

  useEffect(() => {
    void loadPerformanceData();
  }, [loadPerformanceData]);

  useEffect(() => {
    void loadEvaluationData();
  }, [loadEvaluationData]);

  useEffect(() => {
    void loadEvaluationRuntime();
  }, [loadEvaluationRuntime]);

  useEffect(() => {
    void loadQueryCacheRuntime();
  }, [loadQueryCacheRuntime]);

  useEffect(() => {
    const variantNames = Object.keys(evaluationReport?.variants || {}).filter(isVisibleEvaluationVariant);
    if (!variantNames.length) {
      setSelectedVariantName('');
      return;
    }
    setSelectedVariantName((prev) => {
      if (prev && variantNames.includes(prev)) return prev;
      if (evaluationReport?.final_variant && variantNames.includes(evaluationReport.final_variant)) {
        return evaluationReport.final_variant;
      }
      return variantNames[0];
    });
  }, [evaluationReport]);

  useEffect(() => {
    if (!activeEvaluationJob || !['pending', 'running', 'cancelling'].includes(activeEvaluationJob.status)) {
      return;
    }
    const timer = window.setInterval(async () => {
      try {
        const latestJob = await getEvaluationJob(activeEvaluationJob.job_id);
        setActiveEvaluationJob(latestJob);
        setEvaluationJobs((prev) => upsertJob(prev, latestJob));
        if (latestJob.status === 'completed') {
          window.clearInterval(timer);
          void loadEvaluationRuntime();
          void loadQueryCacheRuntime();
          if (latestJob.report_id) {
            void loadEvaluationData(latestJob.report_id);
          } else {
            void loadEvaluationData();
          }
        } else if (latestJob.status === 'failed') {
          window.clearInterval(timer);
          setEvaluationActionError(latestJob.error || '评测任务失败');
          void loadEvaluationRuntime();
          void loadQueryCacheRuntime();
        } else if (latestJob.status === 'cancelled') {
          window.clearInterval(timer);
          setEvaluationActionMessage('评测任务已取消。');
          void loadEvaluationRuntime();
          void loadQueryCacheRuntime();
        }
      } catch (err: unknown) {
        window.clearInterval(timer);
        setEvaluationActionError(err instanceof Error && err.message ? err.message : '轮询评测任务失败');
      }
    }, 2500);
    return () => window.clearInterval(timer);
  }, [activeEvaluationJob, loadEvaluationData, loadEvaluationRuntime, loadQueryCacheRuntime]);

  const trendData = useMemo(
    () =>
      timeSeries.map((point) => ({
        period: new Date(point.period).toLocaleString('zh-CN', {
          month: 'short',
          day: 'numeric',
          ...(granularity === 'hour' ? { hour: '2-digit' as const } : {}),
        }),
        avgDuration: point.avg_total_duration_ms ? Number((point.avg_total_duration_ms / 1000).toFixed(2)) : 0,
        p50Duration: point.p50_total_duration_ms ? Number((point.p50_total_duration_ms / 1000).toFixed(2)) : 0,
        p95Duration: point.p95_total_duration_ms ? Number((point.p95_total_duration_ms / 1000).toFixed(2)) : 0,
      })),
    [timeSeries, granularity],
  );

  const stageChartData = useMemo(
    () =>
      stages.map((stage) => ({
        ...stage,
        stage_label: formatStageLabel(stage.stage),
      })),
    [stages],
  );

  const variantRows = useMemo(
    () =>
      Object.entries(evaluationReport?.variants || {}).filter(([variantName]) =>
        isVisibleEvaluationVariant(variantName),
      ),
    [evaluationReport],
  );
  const activeVariantName =
    selectedVariantName ||
    (evaluationReport?.final_variant && isVisibleEvaluationVariant(evaluationReport.final_variant)
      ? evaluationReport.final_variant
      : '') ||
    variantRows[0]?.[0] ||
    '';
  const selectedVariantPayload = activeVariantName
    ? evaluationReport?.variants?.[activeVariantName]
    : undefined;
  const selectedSummary = selectedVariantPayload?.summary ?? null;
  const finalVariantPayload =
    evaluationReport?.final_variant && isVisibleEvaluationVariant(evaluationReport.final_variant)
      ? evaluationReport.variants?.[evaluationReport.final_variant]
      : undefined;
  const selectedVariantQueryTypes = useMemo(
    () => Object.entries(selectedVariantPayload?.by_query_type || {}),
    [selectedVariantPayload],
  );
  const comparisonRows = useMemo(
    () =>
      Object.entries(evaluationReport?.comparisons || {}).filter(
        ([, comparison]) =>
          isVisibleEvaluationVariant(comparison.variant) &&
          isVisibleEvaluationVariant(comparison.compare_to),
      ),
    [evaluationReport],
  );
  const selectedComparisonRows = useMemo(() => {
    if (!activeVariantName) return comparisonRows;
    const matched = comparisonRows.filter(([, comparison]) =>
      comparison.variant === activeVariantName || comparison.compare_to === activeVariantName,
    );
    return matched.sort((left, right) => {
      const [leftName, leftComparison] = left;
      const [rightName, rightComparison] = right;
      const leftPriority =
        leftComparison.variant === activeVariantName ? 0 : leftComparison.compare_to === activeVariantName ? 1 : 2;
      const rightPriority =
        rightComparison.variant === activeVariantName ? 0 : rightComparison.compare_to === activeVariantName ? 1 : 2;
      if (leftPriority !== rightPriority) return leftPriority - rightPriority;
      return leftName.localeCompare(rightName);
    });
  }, [activeVariantName, comparisonRows]);
  const selectedRagasErrors = useMemo(
    () => Object.entries(selectedSummary?.ragas_errors || {}).filter(([, message]) => Boolean(message)),
    [selectedSummary],
  );
  const preferredTemplateDatasetPath = useMemo(
    () => resolveTemplateDatasetPath(evaluationConfig?.template_dataset_path),
    [evaluationConfig?.template_dataset_path],
  );
  const datasetCatalog = useMemo(() => evaluationConfig?.dataset_catalog || [], [evaluationConfig?.dataset_catalog]);
  const selectedDatasetOption = useMemo(() => {
    const currentPath = datasetPath.trim();
    return datasetCatalog.find((item) => item.path === currentPath) || null;
  }, [datasetCatalog, datasetPath]);
  const isBusinessBenchmarkReport = Boolean(
    evaluationReport?.dataset_name === 'business_benchmark_30' ||
      evaluationReport?.dataset_name === 'business_benchmark_100' ||
      evaluationReport?.dataset_path?.endsWith('/business_benchmark_30.docs.json') ||
      evaluationReport?.dataset_path?.endsWith('/business_benchmark_100.docs.json'),
  );
  const selectedVariantLabel = selectedVariantPayload
    ? formatVariantLabel(activeVariantName, selectedVariantPayload.technique)
    : '未选择方案';
  const selectedVariantCompactLabel = selectedVariantPayload
    ? formatCompactVariantLabel(activeVariantName, selectedVariantPayload)
    : '未选择方案';
  const finalVariantLabel =
    evaluationReport?.final_variant && isVisibleEvaluationVariant(evaluationReport.final_variant)
      ? formatVariantLabel(evaluationReport.final_variant, finalVariantPayload?.technique)
      : '-';
  const finalVariantCompactLabel =
    evaluationReport?.final_variant && isVisibleEvaluationVariant(evaluationReport.final_variant)
      ? formatCompactVariantLabel(evaluationReport.final_variant, finalVariantPayload)
      : '-';
  const currentEvaluationDatasetPath = (
    evaluationMeta?.dataset_path ||
    evaluationReport?.dataset_path ||
    datasetPath
  ).trim();
  const activeDatasetSyncResult = useMemo(() => {
    if (!datasetSyncResult || !currentEvaluationDatasetPath) return null;
    return datasetSyncResult.output_path === currentEvaluationDatasetPath ||
      datasetSyncResult.dataset_path === currentEvaluationDatasetPath
      ? datasetSyncResult
      : null;
  }, [currentEvaluationDatasetPath, datasetSyncResult]);
  const selectedWarnings = useMemo(
    () =>
      (
        activeDatasetSyncResult?.warnings ||
        selectedSummary?.warnings ||
        []
      ).filter((message) => Boolean(message?.trim())),
    [activeDatasetSyncResult, selectedSummary],
  );
  const selectedHeadlineMetricKeys = HEADLINE_METRIC_KEYS;
  const selectedGroundTruthSummary =
    activeDatasetSyncResult?.ground_truth_summary || selectedSummary?.retrieval_ground_truth || null;
  const showDatasetSyncButton = Boolean(
    (activeDatasetSyncResult?.stale_declared_cases_before || 0) > 0 ||
    (selectedSummary?.retrieval_ground_truth?.stale_declared_cases || 0) > 0 ||
    (selectedGroundTruthSummary?.item_name_filter_miss_cases || 0) > 0 ||
    (selectedGroundTruthSummary?.unresolved_cases || 0) > 0,
  );
  const showKnowledgeBaseMigrationButton = Boolean(selectedGroundTruthSummary || currentEvaluationDatasetPath);
  const variantTechniqueMap = useMemo(
    () =>
      Object.fromEntries(
        (evaluationConfig?.variant_catalog || [])
          .filter((variant) => isVisibleEvaluationVariant(variant.name))
          .map((variant) => [variant.name, variant.technique]),
      ),
    [evaluationConfig?.variant_catalog],
  );
  const groupedEvaluationVariants = useMemo(() => {
    const groups = new Map<string, EvaluationVariantOption[]>();
    (evaluationConfig?.variant_catalog || [])
      .filter((variant) => isVisibleEvaluationVariant(variant.name))
      .forEach((variant) => {
        const category = inferVariantCategory(variant);
        groups.set(category, [...(groups.get(category) || []), variant]);
      });

    const orderedCategories = [
      ...VARIANT_CATEGORY_ORDER,
      ...Array.from(groups.keys()).filter((category) => !VARIANT_CATEGORY_ORDER.includes(category)),
    ];
    return orderedCategories
      .filter((category) => (groups.get(category) || []).length > 0)
      .map((category) => ({
        category,
        label: VARIANT_CATEGORY_LABELS[category] || category,
        description: VARIANT_CATEGORY_DESCRIPTIONS[category] || '',
        variants: groups.get(category) || [],
      }));
  }, [evaluationConfig?.variant_catalog]);
  const queryCacheNamespaceRows = useMemo(
    () => Object.entries(queryCacheStats?.namespaces || {}),
    [queryCacheStats],
  );
  const trialVariantLabel =
    trialComboLabel.trim() || formatFeatureComboLabel(trialFeatureKeys, evaluationFeatures);
  const groupedEvaluationFeatures = useMemo(() => {
    const groups = new Map<string, EvaluationFeatureOption[]>();
    evaluationFeatures.forEach((feature) => {
      groups.set(feature.category, [...(groups.get(feature.category) || []), feature]);
    });
    return [
      ...FEATURE_CATEGORY_ORDER,
      ...Array.from(groups.keys()).filter((category) => !FEATURE_CATEGORY_ORDER.includes(category)),
    ]
      .filter((category) => (groups.get(category) || []).length > 0)
      .map((category) => ({
        category,
        label: FEATURE_CATEGORY_LABELS[category] || category,
        features: groups.get(category) || [],
      }));
  }, [evaluationFeatures]);
  const resolvedDraftFeatures = useMemo(
    () => resolveFeatureKeys(draftFeatureKeys, evaluationFeatures),
    [draftFeatureKeys, evaluationFeatures],
  );
  const autoDraftFeatures = resolvedDraftFeatures.filter((key) => !draftFeatureKeys.includes(key));
  const resolvedAppendFeatures = useMemo(
    () => resolveFeatureKeys(appendFeatureKeys, evaluationFeatures),
    [appendFeatureKeys, evaluationFeatures],
  );
  const autoAppendFeatures = resolvedAppendFeatures.filter((key) => !appendFeatureKeys.includes(key));
  const comboFeatureLabelMap = featureLabelMap(evaluationFeatures);
  const activeMetricGroup =
    EVALUATION_METRIC_GROUPS.find((group) => group.key === activeMetricGroupKey) ||
    EVALUATION_METRIC_GROUPS[0];
  const trialMetadata = trialResult?.metadata || null;
  const trialStageRows = useMemo(
    () => Object.entries(trialResult?.stage_durations_ms || {}),
    [trialResult?.stage_durations_ms],
  );
  const isEvaluationBusy =
    evaluationLoading ||
    startingEvaluation ||
    appendingEvaluation ||
    syncingEvaluationDataset ||
    migratingKnowledgeBase ||
    ['pending', 'running', 'cancelling'].includes(activeEvaluationJob?.status || '');

  return (
    <div className="flex-1 overflow-y-auto p-4 md:p-6">
      <div className="mx-auto flex max-w-6xl flex-col gap-6">
        <section className="rounded-2xl border border-gray-200 bg-white p-5 shadow-sm dark:border-gray-800 dark:bg-gray-900/60">
          <div className="flex flex-col gap-5 xl:flex-row xl:items-end xl:justify-between">
            <div className="space-y-4">
              <div>
                <p className="text-xs font-semibold uppercase tracking-[0.2em] text-violet-500">分析看板</p>
                <h1 className="mt-1 flex items-center gap-2 text-2xl font-bold text-gray-900 dark:text-white">
                  {activeTab === 'runtime' ? (
                    <BarChart3 className="h-6 w-6 text-violet-500" />
                  ) : (
                    <Target className="h-6 w-6 text-violet-500" />
                  )}
                  性能分析与量化评测
                </h1>
                <p className="text-sm text-gray-500">
                  {activeTab === 'runtime'
                    ? '查看端到端耗时趋势、阶段分布和运行明细。'
                    : '运行统一评测任务，并查看 RAGAS、检索命中率与时延指标。'}
                </p>
              </div>

              <div className="inline-flex w-full flex-wrap gap-2 rounded-2xl border border-gray-200 bg-gray-50 p-1 dark:border-gray-800 dark:bg-gray-950/50">
                {[
                  { key: 'runtime', label: '运行时性能分析', icon: <BarChart3 className="h-4 w-4" /> },
                  { key: 'evaluation', label: '量化评测', icon: <Target className="h-4 w-4" /> },
                ].map((tab) => {
                  const isActive = activeTab === tab.key;
                  return (
                    <button
                      key={tab.key}
                      type="button"
                      onClick={() => setActiveTab(tab.key as 'runtime' | 'evaluation')}
                      className={`inline-flex items-center gap-2 rounded-xl px-4 py-2 text-sm font-medium transition-colors ${isActive
                          ? 'bg-white text-violet-700 shadow-sm dark:bg-gray-900 dark:text-violet-300'
                          : 'text-gray-600 hover:bg-white/70 hover:text-gray-900 dark:text-gray-400 dark:hover:bg-gray-900/70 dark:hover:text-gray-100'
                        }`}
                    >
                      {tab.icon}
                      {tab.label}
                    </button>
                  );
                })}
              </div>
            </div>

            <div className="flex flex-wrap items-center gap-2">
              {activeTab === 'runtime' ? (
                <>
                  <select
                    value={granularity}
                    onChange={(e) => setGranularity(e.target.value as 'day' | 'hour')}
                    className="rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm dark:border-gray-700 dark:bg-gray-900"
                  >
                    <option value="day">按天</option>
                    <option value="hour">按小时</option>
                  </select>
                  <input
                    type="datetime-local"
                    value={startDate}
                    onChange={(e) => setStartDate(e.target.value)}
                    className="rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm dark:border-gray-700 dark:bg-gray-900"
                  />
                  <input
                    type="datetime-local"
                    value={endDate}
                    onChange={(e) => setEndDate(e.target.value)}
                    className="rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm dark:border-gray-700 dark:bg-gray-900"
                  />
                </>
              ) : (
                <div className="flex min-w-[260px] items-end gap-2">
                  <div className="flex min-w-[260px] flex-col gap-1">
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      历史评测报告（时间 + 对比变量）
                    </span>
                    <select
                      value={selectedReportId}
                      onChange={(e) => void handleReportChange(e.target.value)}
                      disabled={evaluationReports.length === 0 || evaluationLoading}
                      className="min-w-[260px] rounded-lg border border-gray-300 bg-white px-3 py-2 text-sm dark:border-gray-700 dark:bg-gray-900"
                    >
                      {evaluationReports.length === 0 ? (
                        <option value="">暂无评测报告</option>
                      ) : (
                        evaluationReports.map((report) => (
                          <option key={report.report_id} value={report.report_id}>
                            {formatReportTimestamp(report.generated_at)} ·{' '}
                            {formatReportVariantSummary(report, variantTechniqueMap)}
                          </option>
                        ))
                      )}
                    </select>
                  </div>
                  <button
                    type="button"
                    onClick={() => void handleDeleteSelectedReport()}
                    disabled={!selectedReportId || evaluationLoading}
                    className="rounded-lg border border-red-200 p-2 text-red-600 hover:bg-red-50 disabled:cursor-not-allowed disabled:opacity-60 dark:border-red-900/40 dark:text-red-300 dark:hover:bg-red-950/20"
                    title="删除当前评测报告"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
              )}

              <button
                onClick={handleRefresh}
                className="rounded-lg border border-gray-200 p-2 text-gray-600 hover:bg-gray-100 dark:border-gray-700 dark:text-gray-300 dark:hover:bg-gray-800"
                title={activeTab === 'runtime' ? '刷新运行时性能分析' : '刷新量化评测'}
              >
                <RefreshCcw className={`h-4 w-4 ${(activeTab === 'runtime' ? performanceLoading : isEvaluationBusy) ? 'animate-spin' : ''}`} />
              </button>
            </div>
          </div>
        </section>

        {activeTab === 'runtime' ? (
          <>
            <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
              <StatCard
                title="运行次数"
                value={`${summary?.run_count || 0}`}
                icon={<Activity className="h-5 w-5" />}
                loading={performanceLoading && !summary}
              />
              <StatCard
                title="平均总耗时"
                value={formatMs(summary?.avg_total_duration_ms)}
                icon={<Clock3 className="h-5 w-5" />}
                loading={performanceLoading && !summary}
              />
              <StatCard
                title="P50 总耗时"
                value={formatMs(summary?.p50_total_duration_ms)}
                icon={<Search className="h-5 w-5" />}
                loading={performanceLoading && !summary}
              />
              <StatCard
                title="P95 总耗时"
                value={formatMs(summary?.p95_total_duration_ms)}
                icon={<Sparkles className="h-5 w-5" />}
                loading={performanceLoading && !summary}
              />
              <StatCard
                title="平均首 token"
                value={formatMs(summary?.avg_first_token_ms)}
                icon={<Sparkles className="h-5 w-5" />}
                loading={performanceLoading && !summary}
              />
              <StatCard
                title="P50 首 token"
                value={formatMs(summary?.p50_first_token_ms)}
                icon={<Target className="h-5 w-5" />}
                loading={performanceLoading && !summary}
              />
              <StatCard
                title="P95 首 token"
                value={formatMs(summary?.p95_first_token_ms)}
                icon={<ShieldAlert className="h-5 w-5" />}
                loading={performanceLoading && !summary}
              />
              <StatCard
                title="平均首答耗时"
                value={formatMs(summary?.avg_first_answer_ms)}
                subtitle={
                  summary?.p95_first_answer_ms != null
                    ? `P95 ${formatMs(summary.p95_first_answer_ms)}`
                    : undefined
                }
                icon={<FileText className="h-5 w-5" />}
                loading={performanceLoading && !summary}
              />
            </div>

            {performanceError ? (
              <SectionMessage
                title="性能数据加载失败"
                message={performanceError}
                icon={<ShieldAlert className="h-6 w-6" />}
              />
            ) : null}

            {performanceLoading && trendData.length === 0 ? (
              <LoadingSection title="耗时趋势" icon={<Clock3 className="h-5 w-5 text-violet-500" />} />
            ) : trendData.length > 0 ? (
              <section className="rounded-2xl border border-gray-200 bg-white p-4 shadow-sm dark:border-gray-800 dark:bg-gray-900/60">
                <div className="mb-4 flex items-center gap-2">
                  <Clock3 className="h-5 w-5 text-violet-500" />
                  <h2 className="text-lg font-semibold">耗时趋势</h2>
                </div>
                <div className="h-[320px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={trendData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis dataKey="period" tick={{ fontSize: 11 }} />
                      <YAxis tick={{ fontSize: 11 }} />
                      <Tooltip />
                      <Legend />
                      <Line type="monotone" dataKey="avgDuration" stroke="#8b5cf6" name="平均总耗时(s)" strokeWidth={2} />
                      <Line type="monotone" dataKey="p50Duration" stroke="#0f766e" name="P50 总耗时(s)" strokeWidth={2} />
                      <Line type="monotone" dataKey="p95Duration" stroke="#2563eb" name="P95 总耗时(s)" strokeWidth={2} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </section>
            ) : null}

            {performanceLoading && stageChartData.length === 0 ? (
              <LoadingSection title="阶段耗时分布" icon={<Activity className="h-5 w-5 text-violet-500" />} />
            ) : stageChartData.length > 0 ? (
              <section className="rounded-2xl border border-gray-200 bg-white p-4 shadow-sm dark:border-gray-800 dark:bg-gray-900/60">
                <div className="mb-4 flex items-center gap-2">
                  <Activity className="h-5 w-5 text-violet-500" />
                  <h2 className="text-lg font-semibold">阶段耗时分布</h2>
                </div>
                <div className="h-[400px]">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={stageChartData} layout="vertical" margin={{ left: 40 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis type="number" tick={{ fontSize: 11 }} />
                      <YAxis type="category" dataKey="stage_label" tick={{ fontSize: 11 }} width={130} />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="avg_duration_ms" fill="#8b5cf6" name="平均耗时(ms)" />
                      <Bar dataKey="p50_duration_ms" fill="#0f766e" name="P50耗时(ms)" />
                      <Bar dataKey="p95_duration_ms" fill="#2563eb" name="P95耗时(ms)" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </section>
            ) : null}

            <section className="rounded-2xl border border-gray-200 bg-white p-4 shadow-sm dark:border-gray-800 dark:bg-gray-900/60">
              <div className="mb-4 flex items-center gap-2">
                <Search className="h-5 w-5 text-violet-500" />
                <h2 className="text-lg font-semibold">阶段明细</h2>
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-left text-sm">
                  <thead className="border-b border-gray-200 text-gray-500 dark:border-gray-800 dark:text-gray-400">
                    <tr>
                      <th className="px-3 py-2 font-medium">阶段</th>
                      <th className="px-3 py-2 text-right font-medium">次数</th>
                      <th className="px-3 py-2 text-right font-medium">平均耗时</th>
                      <th className="px-3 py-2 text-right font-medium">P50</th>
                      <th className="px-3 py-2 text-right font-medium">P95</th>
                      <th className="px-3 py-2 text-right font-medium">错误率</th>
                    </tr>
                  </thead>
                  <tbody>
                    {performanceLoading && stages.length === 0 ? (
                      <tr>
                        <td colSpan={6} className="px-3 py-8 text-center text-sm text-gray-500">
                          正在加载阶段明细...
                        </td>
                      </tr>
                    ) : stages.length > 0 ? (
                      stages.map((stage) => (
                        <tr key={stage.stage} className="border-b border-gray-100 dark:border-gray-800/70">
                          <td className="px-3 py-3 font-medium text-gray-900 dark:text-white">
                            {formatStageLabel(stage.stage)}
                          </td>
                          <td className="px-3 py-3 text-right">{stage.count}</td>
                          <td className="px-3 py-3 text-right">{formatMs(stage.avg_duration_ms)}</td>
                          <td className="px-3 py-3 text-right">{formatMs(stage.p50_duration_ms)}</td>
                          <td className="px-3 py-3 text-right">{formatMs(stage.p95_duration_ms)}</td>
                          <td className="px-3 py-3 text-right">{(stage.error_rate * 100).toFixed(1)}%</td>
                        </tr>
                      ))
                    ) : (
                      <tr>
                        <td colSpan={6} className="px-3 py-8 text-center text-sm text-gray-500">
                          暂无阶段数据
                        </td>
                      </tr>
                    )}
                  </tbody>
                </table>
              </div>
            </section>
          </>
        ) : null}

        {activeTab === 'evaluation' ? (
          <section className="rounded-2xl border border-gray-200 bg-white p-4 shadow-sm dark:border-gray-800 dark:bg-gray-900/60">
            <div className="flex flex-col gap-4">
              <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <Target className="h-5 w-5 text-violet-500" />
                    <h2 className="text-lg font-semibold">量化评测</h2>
                  </div>
                  <p className="mt-1 text-sm text-gray-500">展示统一评测报告中的 RAGAS、检索命中率和时延指标</p>
                </div>
              </div>

              <div className="inline-flex w-full flex-wrap gap-1 rounded-2xl border border-gray-200 bg-gray-50 p-1 dark:border-gray-800 dark:bg-gray-950/50">
                {[
                  { key: 'run', label: '运行评测', icon: <Play className="h-4 w-4" /> },
                  { key: 'report', label: '报告对比', icon: <Target className="h-4 w-4" /> },
                  { key: 'trial', label: '在线试跑', icon: <Search className="h-4 w-4" /> },
                  { key: 'cache', label: '缓存维护', icon: <Database className="h-4 w-4" /> },
                ].map((tab) => {
                  const isActive = evaluationSubTab === tab.key;
                  return (
                    <button
                      key={tab.key}
                      type="button"
                      onClick={() => setEvaluationSubTab(tab.key as 'run' | 'report' | 'trial' | 'cache')}
                      className={`inline-flex items-center gap-2 rounded-xl px-4 py-2 text-sm font-medium transition-colors ${isActive
                          ? 'bg-white text-violet-700 shadow-sm dark:bg-gray-900 dark:text-violet-300'
                          : 'text-gray-600 hover:bg-white/70 hover:text-gray-900 dark:text-gray-400 dark:hover:bg-gray-900/70 dark:hover:text-gray-100'
                        }`}
                    >
                      {tab.icon}
                      {tab.label}
                    </button>
                  );
                })}
              </div>

              {evaluationSubTab !== 'report' ? (
              <div className="rounded-2xl border border-gray-200 bg-gray-50/70 p-4 dark:border-gray-800 dark:bg-gray-950/40">
                <div className="flex flex-col gap-4">
                  <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                    <div>
                      <h3 className="flex items-center gap-2 text-base font-semibold text-gray-900 dark:text-white">
                        {evaluationSubTab === 'trial' ? (
                          <Search className="h-4 w-4 text-violet-500" />
                        ) : evaluationSubTab === 'cache' ? (
                          <Database className="h-4 w-4 text-violet-500" />
                        ) : (
                          <Play className="h-4 w-4 text-violet-500" />
                        )}
                        {evaluationSubTab === 'trial' ? '方案在线试跑' : evaluationSubTab === 'cache' ? '缓存维护' : '运行统一评测'}
                      </h3>
                      <p className="mt-1 text-sm text-gray-500">
                        {evaluationSubTab === 'trial'
                          ? '选择功能组合并输入单条问题，检查回答、时延和检索计划。'
                          : evaluationSubTab === 'cache'
                            ? '查看缓存命中情况；含 Cache 的组合会先重置并预热一轮再正式统计。'
                            : '填写评测数据集路径，添加要对比的功能组合，系统会在后台执行并自动生成报告。'}
                      </p>
                    </div>
                    <button
                      onClick={() => setDatasetPath(preferredTemplateDatasetPath)}
                      className="rounded-lg border border-gray-200 px-3 py-2 text-sm text-gray-600 hover:bg-gray-100 dark:border-gray-700 dark:text-gray-300 dark:hover:bg-gray-800"
                      type="button"
                    >
                      使用模板路径
                    </button>
                  </div>

                  <div className={evaluationSubTab === 'run' ? 'grid grid-cols-1 gap-4 xl:grid-cols-[1.3fr_1fr]' : 'hidden'}>
                    <div className="space-y-4">
                      <div>
                        <label className="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">评测集</label>
                        {datasetCatalog.length > 0 ? (
                          <select
                            value={selectedDatasetOption?.key || 'custom'}
                            onChange={(event) => {
                              const item = datasetCatalog.find((option) => option.key === event.target.value);
                              if (item) setDatasetPath(item.path);
                            }}
                            className="mb-3 w-full rounded-xl border border-gray-300 bg-white px-3 py-2.5 text-sm dark:border-gray-700 dark:bg-gray-900"
                          >
                            {datasetCatalog.map((item) => (
                              <option key={item.key} value={item.key}>
                                {item.label} · {item.case_count} 条{item.exists ? '' : ' · 文件缺失'}
                              </option>
                            ))}
                            <option value="custom">自定义路径</option>
                          </select>
                        ) : null}
                        <label className="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">数据集路径</label>
                        <input
                          type="text"
                          value={datasetPath}
                          onChange={(e) => setDatasetPath(e.target.value)}
                          placeholder={preferredTemplateDatasetPath || '请输入 JSON / JSONL 数据集路径'}
                          className="w-full rounded-xl border border-gray-300 bg-white px-3 py-2.5 text-sm dark:border-gray-700 dark:bg-gray-900"
                        />
                        <p className="mt-2 text-xs text-gray-500">
                          {selectedDatasetOption
                            ? `${selectedDatasetOption.dataset_name}：${selectedDatasetOption.description}`
                            : `模板：${preferredTemplateDatasetPath}`}
                        </p>
                      </div>

                        <div>
                          <div className="mb-2 flex items-center justify-between gap-2">
                            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">对比项</label>
                            <button
                              type="button"
                              onClick={handleStartAddCombo}
                              disabled={runComparisonCount >= MAX_FEATURE_COMPARISONS || evaluationFeatures.length === 0}
                              className="inline-flex items-center gap-1 text-xs text-violet-600 hover:text-violet-700 disabled:cursor-not-allowed disabled:opacity-50 dark:text-violet-400"
                            >
                              <Plus className="h-3.5 w-3.5" />
                              添加对比项
                            </button>
                          </div>
                        <div className="space-y-4">
                          <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
                            <label
                              className={`flex cursor-pointer items-start gap-3 rounded-xl border p-3 transition-colors ${includeBaseline
                                  ? 'border-violet-300 bg-violet-50/70 dark:border-violet-700 dark:bg-violet-950/20'
                                  : 'border-gray-200 bg-white dark:border-gray-800 dark:bg-gray-900/70'
                                }`}
                            >
                              <input
                                type="checkbox"
                                checked={includeBaseline}
                                disabled={!includeBaseline && runComparisonCount >= MAX_FEATURE_COMPARISONS}
                                onChange={(event) => setIncludeBaseline(event.target.checked)}
                                className="mt-1 h-4 w-4 rounded border-gray-300 text-violet-600 disabled:cursor-not-allowed"
                              />
                              <div className="min-w-0">
                                <div className="text-sm font-semibold text-gray-900 dark:text-white">Baseline</div>
                                <div className="mt-2 flex flex-wrap gap-1">
                                  <span className="rounded-full bg-gray-100 px-2 py-0.5 text-[11px] text-gray-600 dark:bg-gray-800 dark:text-gray-300">
                                    embedding + rerank + answer
                                  </span>
                                </div>
                              </div>
                            </label>
                            {featureCombos.map((combo) => {
                              const resolved = resolveFeatureKeys(combo.features, evaluationFeatures);
                              const labels = resolved.map((key) => comboFeatureLabelMap.get(key) || key);
                              return (
                                <div
                                  key={combo.id}
                                  className="rounded-xl border border-violet-200 bg-violet-50/70 p-3 dark:border-violet-900 dark:bg-violet-950/20"
                                >
                                  <div className="flex items-start justify-between gap-3">
                                    <div className="min-w-0">
                                      <div className="text-sm font-semibold text-gray-900 dark:text-white">{combo.label}</div>
                                      <div className="mt-2 flex flex-wrap gap-1">
                                        {labels.length ? labels.map((label) => (
                                          <span key={label} className="rounded-full bg-white px-2 py-0.5 text-[11px] text-violet-700 dark:bg-gray-900 dark:text-violet-300">
                                            {label}
                                          </span>
                                        )) : (
                                          <span className="rounded-full bg-gray-100 px-2 py-0.5 text-[11px] text-gray-600 dark:bg-gray-800 dark:text-gray-300">
                                            embedding + rerank + answer
                                          </span>
                                        )}
                                      </div>
                                    </div>
                                    <div className="flex shrink-0 items-center gap-1">
                                      <button
                                        type="button"
                                        onClick={() => handleEditCombo(combo)}
                                        className="rounded-lg p-1.5 text-gray-500 hover:bg-white hover:text-violet-600 dark:hover:bg-gray-900"
                                        title="编辑对比项"
                                      >
                                        <Pencil className="h-4 w-4" />
                                      </button>
                                      <button
                                        type="button"
                                        onClick={() => handleDeleteCombo(combo.id)}
                                        className="rounded-lg p-1.5 text-gray-500 hover:bg-white hover:text-red-600 dark:hover:bg-gray-900"
                                        title="删除对比项"
                                      >
                                        <Trash2 className="h-4 w-4" />
                                      </button>
                                    </div>
                                  </div>
                                </div>
                              );
                            })}
                          </div>

                          {editingComboId ? (
                            <section className="rounded-xl border border-violet-200 bg-white p-3 dark:border-violet-900 dark:bg-gray-900/70">
                              <div className="mb-3 flex items-center justify-between gap-2">
                                <div>
                                  <h4 className="text-sm font-semibold text-gray-900 dark:text-white">
                                    {editingComboId === 'new' ? '添加功能组合' : '编辑功能组合'}
                                  </h4>
                                  <p className="mt-1 text-xs text-gray-500">这里只选择增强功能；Baseline 在上方作为独立对比项勾选。</p>
                                </div>
                                <button
                                  type="button"
                                  onClick={() => setEditingComboId(null)}
                                  className="rounded-lg p-1.5 text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800"
                                  title="关闭"
                                >
                                  <X className="h-4 w-4" />
                                </button>
                              </div>
                              <input
                                type="text"
                                value={draftComboLabel}
                                onChange={(event) => setDraftComboLabel(event.target.value)}
                                placeholder={formatFeatureComboLabel(draftFeatureKeys, evaluationFeatures)}
                                className="mb-3 w-full rounded-xl border border-gray-300 bg-white px-3 py-2.5 text-sm dark:border-gray-700 dark:bg-gray-900"
                              />
                              <div className="space-y-3">
                                {groupedEvaluationFeatures.map((group) => (
                                  <div key={group.category}>
                                    <div className="mb-2 text-xs font-semibold text-gray-500">{group.label}</div>
                                    <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
                                      {group.features.map((feature) => {
                                        const checked = draftFeatureKeys.includes(feature.key);
                                        const autoEnabled = autoDraftFeatures.includes(feature.key);
                                        return (
                                          <label
                                            key={feature.key}
                                            className={`flex cursor-pointer items-start gap-3 rounded-xl border px-3 py-3 transition-colors ${checked || autoEnabled
                                                ? 'border-violet-300 bg-violet-50 dark:border-violet-700 dark:bg-violet-950/20'
                                                : 'border-gray-200 bg-white dark:border-gray-800 dark:bg-gray-950/40'
                                              }`}
                                          >
                                            <input
                                              type="checkbox"
                                              checked={checked}
                                              onChange={() => handleToggleDraftFeature(feature.key)}
                                              className="mt-1 h-4 w-4 rounded border-gray-300 text-violet-600"
                                            />
                                            <div className="min-w-0">
                                              <div className="flex flex-wrap items-center gap-2 text-sm font-medium text-gray-900 dark:text-white">
                                                {feature.label}
                                                {autoEnabled ? (
                                                  <span className="rounded-full bg-violet-100 px-2 py-0.5 text-[11px] text-violet-700 dark:bg-violet-950 dark:text-violet-300">
                                                    自动启用
                                                  </span>
                                                ) : null}
                                              </div>
                                              <div className="mt-1 text-xs text-gray-500">{feature.description}</div>
                                            </div>
                                          </label>
                                        );
                                      })}
                                    </div>
                                  </div>
                                ))}
                              </div>
                              <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
                                <div className="text-xs text-gray-500">
                                  当前组合：{formatFeatureComboLabel(draftFeatureKeys, evaluationFeatures)}
                                </div>
                                <button
                                  type="button"
                                  onClick={handleSaveCombo}
                                  className="inline-flex items-center justify-center gap-2 rounded-xl bg-violet-600 px-3 py-2 text-sm font-medium text-white hover:bg-violet-700"
                                >
                                  保存对比项
                                </button>
                              </div>
                            </section>
                          ) : null}

                          {evaluationFeatures.length === 0 ? (
                            <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-700 dark:border-amber-900/40 dark:bg-amber-950/30 dark:text-amber-300">
                              正在加载功能目录，加载完成后即可添加组合。
                            </div>
                          ) : null}
                        </div>
                      </div>
                    </div>

                    <div className="rounded-xl border border-gray-200 bg-white p-4 dark:border-gray-800 dark:bg-gray-900">
                      <div className="flex items-center justify-between gap-2">
                        <div>
                          <p className="text-sm font-medium text-gray-900 dark:text-white">任务状态</p>
                          <p className="mt-1 text-xs text-gray-500">当前前端会轮询后台评测任务状态</p>
                        </div>
                        {activeEvaluationJob ? (
                          <span
                            className={`rounded-full px-2.5 py-1 text-xs font-medium ${activeEvaluationJob.status === 'completed'
                                ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300'
                                : activeEvaluationJob.status === 'cancelled'
                                  ? 'bg-gray-200 text-gray-700 dark:bg-gray-800 dark:text-gray-300'
                                : activeEvaluationJob.status === 'failed'
                                  ? 'bg-red-100 text-red-700 dark:bg-red-950/40 dark:text-red-300'
                                  : 'bg-amber-100 text-amber-700 dark:bg-amber-950/40 dark:text-amber-300'
                              }`}
                          >
                            {formatJobStatus(activeEvaluationJob.status)}
                          </span>
                        ) : null}
                      </div>

                      <div className="mt-4 space-y-3 text-sm">
                        <div className="rounded-lg bg-gray-50 px-3 py-2 dark:bg-gray-950/60">
                          <div className="text-xs text-gray-500">当前进度</div>
                          <div className="mt-1 font-medium text-gray-900 dark:text-white">
                            {activeEvaluationJob?.progress_message || '还没有启动评测任务'}
                          </div>
                        </div>
                        {activeEvaluationJob ? (
                          <>
                            <div className="flex items-center justify-between gap-3 text-xs text-gray-500">
                              <span>任务 ID</span>
                              <span className="font-mono text-[11px] text-gray-700 dark:text-gray-300">{activeEvaluationJob.job_id}</span>
                            </div>
                            <div className="flex items-center justify-between gap-3 text-xs text-gray-500">
                              <span>已完成变体</span>
                              <span className="text-gray-700 dark:text-gray-300">
                                {activeEvaluationJob.completed_variants}/{activeEvaluationJob.total_variants}
                              </span>
                            </div>
                            <div className="flex items-center justify-between gap-3 text-xs text-gray-500">
                              <span>样本数</span>
                              <span className="text-gray-700 dark:text-gray-300">{activeEvaluationJob.case_count || '-'}</span>
                            </div>
                            {activeEvaluationJob.phase ? (
                              <div className="flex items-center justify-between gap-3 text-xs text-gray-500">
                                <span>当前阶段</span>
                                <span className="text-gray-700 dark:text-gray-300">
                                  {formatEvaluationPhase(activeEvaluationJob.phase)}
                                  {activeEvaluationJob.phase === 'warmup' && activeEvaluationJob.warmup_rounds
                                    ? `（${activeEvaluationJob.warmup_round || 0}/${activeEvaluationJob.warmup_rounds}）`
                                    : ''}
                                </span>
                              </div>
                            ) : null}
                            {activeEvaluationJob.current_variant_total_cases ? (
                              <div className="flex items-center justify-between gap-3 text-xs text-gray-500">
                                <span>{activeEvaluationJob.phase === 'warmup' ? '缓存预热进度' : '当前变体样本进度'}</span>
                                <span className="text-gray-700 dark:text-gray-300">
                                  {activeEvaluationJob.completed_cases || 0}/{activeEvaluationJob.current_variant_total_cases}
                                </span>
                              </div>
                            ) : null}
                            {activeEvaluationJob.current_case_id || activeEvaluationJob.current_case_query ? (
                              <div className="rounded-lg bg-gray-50 px-3 py-2 text-xs text-gray-500 dark:bg-gray-950/60">
                                <div>当前样本</div>
                                <div className="mt-1 text-gray-700 dark:text-gray-300">
                                  {activeEvaluationJob.current_case_id || activeEvaluationJob.current_case_query}
                                </div>
                              </div>
                            ) : null}
                            {activeEvaluationJob.report_id ? (
                              <div className="flex items-center justify-between gap-3 text-xs text-gray-500">
                                <span>输出报告</span>
                                <span className="text-gray-700 dark:text-gray-300">{activeEvaluationJob.report_id}</span>
                              </div>
                            ) : null}
                            {activeEvaluationJob.last_progress_at ? (
                              <div className="flex items-center justify-between gap-3 text-xs text-gray-500">
                                <span>最后进展</span>
                                <span className="text-gray-700 dark:text-gray-300">
                                  {formatReportTimestamp(activeEvaluationJob.last_progress_at)}
                                </span>
                              </div>
                            ) : null}
                          </>
                        ) : null}
                      </div>

                      <div className="mt-4 grid grid-cols-1 gap-2 sm:grid-cols-2">
                        <button
                          type="button"
                          onClick={() => void handleStartEvaluation()}
                          disabled={startingEvaluation || runComparisonCount === 0 || ['pending', 'running', 'cancelling'].includes(activeEvaluationJob?.status || '')}
                          className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-violet-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-violet-700 disabled:cursor-not-allowed disabled:opacity-60"
                        >
                          {startingEvaluation ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <Play className="h-4 w-4" />
                          )}
                          开始评测
                        </button>
                        <button
                          type="button"
                          onClick={() => void handleCancelEvaluation()}
                          disabled={!['pending', 'running', 'cancelling'].includes(activeEvaluationJob?.status || '')}
                          className="inline-flex w-full items-center justify-center gap-2 rounded-xl border border-amber-300 bg-white px-4 py-2.5 text-sm font-medium text-amber-700 hover:bg-amber-50 disabled:cursor-not-allowed disabled:opacity-60 dark:border-amber-800 dark:bg-gray-950/40 dark:text-amber-300 dark:hover:bg-amber-950/20"
                        >
                          {activeEvaluationJob?.status === 'cancelling' ? (
                            <Loader2 className="h-4 w-4 animate-spin" />
                          ) : (
                            <Square className="h-4 w-4" />
                          )}
                          停止评测
                        </button>
                      </div>
                    </div>
                  </div>

                  {evaluationActionError ? (
                    <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-700 dark:border-amber-900/40 dark:bg-amber-950/30 dark:text-amber-300">
                      {evaluationActionError}
                    </div>
                  ) : null}

                  {evaluationActionMessage ? (
                    <div className="rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-700 dark:border-emerald-900/40 dark:bg-emerald-950/30 dark:text-emerald-300">
                      {evaluationActionMessage}
                    </div>
                  ) : null}

                  <div className={evaluationSubTab === 'trial' ? 'rounded-2xl border border-gray-200 bg-white p-4 dark:border-gray-800 dark:bg-gray-900' : 'hidden'}>
                    <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                      <div>
                        <h3 className="flex items-center gap-2 text-base font-semibold text-gray-900 dark:text-white">
                          <Search className="h-4 w-4 text-violet-500" />
                          方案在线试跑
                        </h3>
                        <p className="mt-1 text-sm text-gray-500">
                          直接针对某个评测方案输入单条问题，查看回答、时延、Router 判定、检索计划和证据摘要。
                        </p>
                      </div>
                    </div>

                    <div className="mt-4 grid grid-cols-1 gap-4 xl:grid-cols-[320px_1fr]">
                      <div className="space-y-4">
                        <div>
                          <label className="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">试跑功能组合</label>
                          <input
                            type="text"
                            value={trialComboLabel}
                            onChange={(event) => setTrialComboLabel(event.target.value)}
                            placeholder={formatFeatureComboLabel(trialFeatureKeys, evaluationFeatures)}
                            className="mb-3 w-full rounded-xl border border-gray-300 bg-white px-3 py-2.5 text-sm dark:border-gray-700 dark:bg-gray-900"
                          />
                          <div className="max-h-[360px] space-y-3 overflow-y-auto pr-1">
                            {groupedEvaluationFeatures.map((group) => (
                              <div key={group.category}>
                                <div className="mb-2 text-xs font-semibold text-gray-500">{group.label}</div>
                                <div className="space-y-2">
                                  {group.features.map((feature) => {
                                    const checked = trialFeatureKeys.includes(feature.key);
                                    return (
                                      <label
                                        key={feature.key}
                                        className={`flex cursor-pointer items-start gap-3 rounded-xl border px-3 py-2.5 ${checked
                                            ? 'border-violet-300 bg-violet-50 dark:border-violet-700 dark:bg-violet-950/20'
                                            : 'border-gray-200 bg-white dark:border-gray-800 dark:bg-gray-950/40'
                                          }`}
                                      >
                                        <input
                                          type="checkbox"
                                          checked={checked}
                                          onChange={() => handleToggleTrialFeature(feature.key)}
                                          className="mt-1 h-4 w-4 rounded border-gray-300 text-violet-600"
                                        />
                                        <span className="text-sm font-medium text-gray-900 dark:text-white">{feature.label}</span>
                                      </label>
                                    );
                                  })}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>

                        <div>
                          <label className="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">测试问题</label>
                          <textarea
                            value={trialQuery}
                            onChange={(e) => setTrialQuery(e.target.value)}
                            placeholder="例如：HAK 180 和 HAK 280 的区别是什么，分别适用于哪些场景？"
                            rows={5}
                            className="w-full rounded-xl border border-gray-300 bg-white px-3 py-2.5 text-sm leading-6 dark:border-gray-700 dark:bg-gray-900"
                          />
                        </div>

                        <button
                          type="button"
                          onClick={() => void handleRunVariantTrial()}
                          disabled={trialLoading || trialFeatureKeys.length === 0}
                          className="inline-flex w-full items-center justify-center gap-2 rounded-xl bg-violet-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-violet-700 disabled:cursor-not-allowed disabled:opacity-60"
                        >
                          {trialLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                          开始试跑
                        </button>

                        {trialError ? (
                          <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-700 dark:border-amber-900/40 dark:bg-amber-950/30 dark:text-amber-300">
                            {trialError}
                          </div>
                        ) : null}
                      </div>

                      <div className="space-y-4">
                        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
                          <StatCard
                            title="当前方案"
                            value={trialVariantLabel}
                            subtitle={trialResult?.technique || '阻塞式单题验证'}
                            icon={<Target className="h-5 w-5" />}
                            loading={trialLoading && !trialResult}
                          />
                          <StatCard
                            title="总耗时"
                            value={formatMs(trialResult?.latency_ms)}
                            subtitle={
                              trialResult?.first_token_ms != null
                                ? `首 token ${formatMs(trialResult.first_token_ms)}`
                                : trialResult?.first_answer_ms != null
                                  ? `首答 ${formatMs(trialResult.first_answer_ms)}`
                                  : undefined
                            }
                            icon={<Clock3 className="h-5 w-5" />}
                            loading={trialLoading && !trialResult}
                          />
                          <StatCard
                            title="复杂度"
                            value={trialMetadata?.query_complexity || '-'}
                            subtitle={trialMetadata?.query_complexity_reason || undefined}
                            icon={<Sparkles className="h-5 w-5" />}
                            loading={trialLoading && !trialResult}
                          />
                          <StatCard
                            title="Router 决策"
                            value={formatRouterDecision(trialMetadata?.router_decision)}
                            subtitle={trialMetadata?.grounded_mode ? '严格 grounded 已启用' : undefined}
                            icon={<Activity className="h-5 w-5" />}
                            loading={trialLoading && !trialResult}
                          />
                        </div>

                        {trialResult ? (
                          <div className="space-y-4">
                            {trialResult.error ? (
                              <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-700 dark:border-amber-900/40 dark:bg-amber-950/30 dark:text-amber-300">
                                {trialResult.error}
                              </div>
                            ) : null}

                            <div className="rounded-2xl border border-gray-200 bg-white p-4 shadow-sm dark:border-gray-800 dark:bg-gray-900/60">
                              <div className="mb-3 flex items-center gap-2">
                                <FileText className="h-4 w-4 text-violet-500" />
                                <h4 className="text-base font-semibold text-gray-900 dark:text-white">回答预览</h4>
                              </div>
                              {trialResult.answer ? (
                                <MarkdownRenderer content={trialResult.answer} />
                              ) : (
                                <p className="text-sm text-gray-500">当前没有返回答案。</p>
                              )}
                            </div>

                            <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
                              <div className="rounded-2xl border border-gray-200 bg-white p-4 shadow-sm dark:border-gray-800 dark:bg-gray-900/60">
                                <div className="mb-3 flex items-center gap-2">
                                  <Activity className="h-4 w-4 text-violet-500" />
                                  <h4 className="text-base font-semibold text-gray-900 dark:text-white">执行元数据</h4>
                                </div>
                                <div className="space-y-2 text-sm text-gray-600 dark:text-gray-300">
                                  <div>问题类型：{trialMetadata?.query_type || '-'}</div>
                                  <div>问题族：{trialMetadata?.router_query_family || '-'}</div>
                                  <div>复杂度：{trialMetadata?.query_complexity || '-'}</div>
                                  <div>Router：{formatRouterDecision(trialMetadata?.router_decision)}</div>
                                  <div>Grounded：{trialMetadata?.grounded_mode ? '已启用' : '未启用'}</div>
                                  <div>CRAG：{trialMetadata?.crag_router_enabled ? '已启用' : '未启用'}</div>
                                  <div>
                                    Anchor 目标：
                                    {trialMetadata?.query_anchor_targets?.length
                                      ? ` ${trialMetadata.query_anchor_targets.join('、')}`
                                      : ' -'}
                                  </div>
                                  <div>
                                    焦点词：
                                    {trialMetadata?.query_focus_terms?.length
                                      ? ` ${trialMetadata.query_focus_terms.join('、')}`
                                      : ' -'}
                                  </div>
                                  <div>
                                    检索计划：
                                    <span className="ml-1 font-mono text-xs text-gray-500 dark:text-gray-400">
                                      {JSON.stringify(trialMetadata?.retrieval_plan || {}, null, 0)}
                                    </span>
                                  </div>
                                  <div>
                                    证据覆盖：
                                    {trialMetadata?.evidence_coverage_summary?.coverage_score != null
                                      ? ` ${(trialMetadata.evidence_coverage_summary.coverage_score * 100).toFixed(1)}%`
                                      : ' -'}
                                  </div>
                                  <div>
                                    目标覆盖：
                                    {toNumericValue(trialMetadata?.target_coverage?.coverage_rate) != null
                                      ? ` ${((toNumericValue(trialMetadata?.target_coverage?.coverage_rate) || 0) * 100).toFixed(1)}%`
                                      : ' -'}
                                  </div>
                                  <div>
                                    Evidence Pack：
                                    {trialMetadata?.evidence_pack_summary
                                      ? ` ${JSON.stringify(trialMetadata.evidence_pack_summary)}`
                                      : ' -'}
                                  </div>
                                  <div>
                                    缓存命中：
                                    {trialMetadata?.cache_summary?.overall
                                      ? ` ${formatMetricValue('cache_hit_rate', toNumericValue(trialMetadata.cache_summary.overall.hit_rate))}`
                                      : ' -'}
                                  </div>
                                </div>
                              </div>

                              <div className="rounded-2xl border border-gray-200 bg-white p-4 shadow-sm dark:border-gray-800 dark:bg-gray-900/60">
                                <div className="mb-3 flex items-center gap-2">
                                  <Clock3 className="h-4 w-4 text-violet-500" />
                                  <h4 className="text-base font-semibold text-gray-900 dark:text-white">阶段耗时</h4>
                                </div>
                                {trialStageRows.length > 0 ? (
                                  <div className="space-y-2 text-sm text-gray-600 dark:text-gray-300">
                                    {trialStageRows.map(([stageName, duration]) => (
                                      <div key={stageName} className="flex items-center justify-between gap-3">
                                        <span>{formatStageLabel(stageName)}</span>
                                        <span className="font-medium text-gray-900 dark:text-white">
                                          {formatMs(duration)}
                                        </span>
                                      </div>
                                    ))}
                                  </div>
                                ) : (
                                  <p className="text-sm text-gray-500">当前没有阶段耗时数据。</p>
                                )}
                              </div>
                            </div>

                            <div className="rounded-2xl border border-gray-200 bg-white p-4 shadow-sm dark:border-gray-800 dark:bg-gray-900/60">
                              <div className="mb-3 flex items-center gap-2">
                                <Database className="h-4 w-4 text-violet-500" />
                                <h4 className="text-base font-semibold text-gray-900 dark:text-white">Top 检索片段预览</h4>
                              </div>
                              {trialResult.retrieved_context_preview.length > 0 ? (
                                <div className="space-y-3">
                                  {trialResult.retrieved_context_preview.map((preview, index) => (
                                    <div key={`${trialResult.retrieved_context_ids[index] || 'ctx'}-${index}`} className="rounded-xl bg-gray-50 px-4 py-3 dark:bg-gray-950/60">
                                      <div className="text-xs text-gray-500">
                                        {(trialResult.retrieved_context_ids[index] || '未提供 chunk_id')}
                                        {trialResult.retrieved_context_titles[index]
                                          ? ` · ${trialResult.retrieved_context_titles[index]}`
                                          : ''}
                                      </div>
                                      <div className="mt-2 text-sm leading-6 text-gray-700 dark:text-gray-200">
                                        {preview}
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              ) : (
                                <p className="text-sm text-gray-500">当前没有检索片段预览。</p>
                              )}
                            </div>
                          </div>
                        ) : (
                          <div className="rounded-2xl border border-dashed border-gray-300 bg-gray-50 px-6 py-8 text-center text-sm text-gray-500 dark:border-gray-700 dark:bg-gray-950/40">
                            选择方案并输入问题后，这里会展示回答、时延、Router 判定和检索片段预览。
                          </div>
                        )}
                      </div>
                    </div>
                  </div>

                  {evaluationSubTab === 'run' && evaluationJobs.length > 0 ? (
                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-sm">
                        <thead className="border-b border-gray-200 text-gray-500 dark:border-gray-800 dark:text-gray-400">
                          <tr>
                            <th className="px-3 py-2 font-medium">最近任务</th>
                            <th className="px-3 py-2 font-medium">状态</th>
                            <th className="px-3 py-2 font-medium">数据集</th>
                            <th className="px-3 py-2 text-right font-medium">进度</th>
                          </tr>
                        </thead>
                        <tbody>
                          {evaluationJobs.slice(0, 5).map((job) => (
                            <tr
                              key={job.job_id}
                              className={`border-b border-gray-100 dark:border-gray-800/70 ${activeEvaluationJob?.job_id === job.job_id ? 'bg-violet-50/70 dark:bg-violet-950/20' : ''
                                }`}
                            >
                              <td className="px-3 py-3">
                                <div className="flex items-center gap-2">
                                  {job.status === 'completed' ? (
                                    <CheckCircle2 className="h-4 w-4 text-emerald-500" />
                                  ) : job.status === 'running' ? (
                                    <Loader2 className="h-4 w-4 animate-spin text-amber-500" />
                                  ) : (
                                    <Target className="h-4 w-4 text-gray-400" />
                                  )}
                                  <button
                                    type="button"
                                    onClick={() => setActiveEvaluationJob(job)}
                                    className="font-mono text-xs text-gray-700 hover:text-violet-600 dark:text-gray-300 dark:hover:text-violet-400"
                                  >
                                    {job.job_id}
                                  </button>
                                </div>
                              </td>
                              <td className="px-3 py-3">{formatJobStatus(job.status)}</td>
                              <td className="px-3 py-3 text-xs text-gray-500">{job.dataset_path}</td>
                              <td className="px-3 py-3 text-right text-xs text-gray-500">
                                {job.completed_variants}/{job.total_variants}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  ) : null}

                  <div className={evaluationSubTab === 'cache' ? 'rounded-2xl border border-gray-200 bg-white p-4 dark:border-gray-800 dark:bg-gray-900' : 'hidden'}>
                    <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                      <div>
                        <h3 className="flex items-center gap-2 text-base font-semibold text-gray-900 dark:text-white">
                          <Database className="h-4 w-4 text-violet-500" />
                          多级缓存运行态
                        </h3>
                        <p className="mt-1 text-sm text-gray-500">
                          查看 L0/L1/L2 命中情况，必要时手动失效缓存后重新跑评测，比较冷缓存与热缓存收益。
                        </p>
                      </div>
                      <button
                        type="button"
                        onClick={() => void handleResetQueryCache()}
                        disabled={resettingQueryCache}
                        className="inline-flex items-center justify-center gap-2 rounded-xl border border-violet-300 bg-white px-3 py-2 text-sm font-medium text-violet-700 hover:bg-violet-50 disabled:cursor-not-allowed disabled:opacity-60 dark:border-violet-800 dark:bg-gray-950/40 dark:text-violet-300 dark:hover:bg-violet-950/20"
                      >
                        {resettingQueryCache ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCcw className="h-4 w-4" />}
                        清空查询缓存
                      </button>
                    </div>

                    {queryCacheError ? (
                      <div className="mt-4 rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-700 dark:border-amber-900/40 dark:bg-amber-950/30 dark:text-amber-300">
                        {queryCacheError}
                      </div>
                    ) : null}

                    {queryCacheActionMessage ? (
                      <div className="mt-4 rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-700 dark:border-emerald-900/40 dark:bg-emerald-950/30 dark:text-emerald-300">
                        {queryCacheActionMessage}
                      </div>
                    ) : null}

                    <div className="mt-4 grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
                      <StatCard
                        title="缓存状态"
                        value={queryCacheStats?.enabled ? '已启用' : '已关闭'}
                        subtitle={queryCacheStats?.redis_enabled ? 'Redis L2 已连接' : '当前使用本地缓存层'}
                        icon={<Database className="h-5 w-5" />}
                        loading={queryCacheLoading && !queryCacheStats}
                      />
                        <StatCard
                          title="全局命中率"
                          value={formatMetricValue('cache_hit_rate', toNumericValue(queryCacheStats?.overall?.hit_rate))}
                          subtitle={`缓存版本 ${queryCacheStats?.epoch ?? '-'}`}
                          icon={<Activity className="h-5 w-5" />}
                          loading={queryCacheLoading && !queryCacheStats}
                        />
                      <StatCard
                        title="L1 条目数"
                        value={formatNumber(queryCacheStats?.l1_size ?? null, 0)}
                        subtitle={`全局写入 ${formatNumber(toNumericValue(queryCacheStats?.overall?.writes), 0)}`}
                        icon={<BarChart3 className="h-5 w-5" />}
                        loading={queryCacheLoading && !queryCacheStats}
                      />
                      <StatCard
                        title="默认缓存域"
                        value={`${queryCacheStats?.default_namespaces?.length || 0}`}
                        subtitle={(queryCacheStats?.default_namespaces || []).map(formatNamespaceLabel).join(' · ') || '未配置'}
                        icon={<Target className="h-5 w-5" />}
                        loading={queryCacheLoading && !queryCacheStats}
                      />
                    </div>

                    <div className="mt-4 overflow-x-auto">
                      <table className="w-full text-left text-sm">
                        <thead className="border-b border-gray-200 text-gray-500 dark:border-gray-800 dark:text-gray-400">
                          <tr>
                            <th className="px-3 py-2 font-medium">命名空间</th>
                            <th className="px-3 py-2 text-right font-medium">Lookups</th>
                            <th className="px-3 py-2 text-right font-medium">Hits</th>
                            <th className="px-3 py-2 text-right font-medium">L0</th>
                            <th className="px-3 py-2 text-right font-medium">L1</th>
                            <th className="px-3 py-2 text-right font-medium">L2</th>
                            <th className="px-3 py-2 text-right font-medium">Misses</th>
                            <th className="px-3 py-2 text-right font-medium">Writes</th>
                            <th className="px-3 py-2 text-right font-medium">命中率</th>
                          </tr>
                        </thead>
                        <tbody>
                          {queryCacheLoading && queryCacheNamespaceRows.length === 0 ? (
                            <tr>
                              <td colSpan={9} className="px-3 py-8 text-center text-sm text-gray-500">
                                正在加载缓存统计...
                              </td>
                            </tr>
                          ) : queryCacheNamespaceRows.length > 0 ? (
                            queryCacheNamespaceRows.map(([namespace, stats]) => (
                              <tr key={namespace} className="border-b border-gray-100 dark:border-gray-800/70">
                                <td className="px-3 py-3 font-medium text-gray-900 dark:text-white">
                                  {formatNamespaceLabel(namespace)}
                                </td>
                                <td className="px-3 py-3 text-right">{formatNumber(toNumericValue(stats.lookups), 0)}</td>
                                <td className="px-3 py-3 text-right">{formatNumber(toNumericValue(stats.hits), 0)}</td>
                                <td className="px-3 py-3 text-right">{formatNumber(toNumericValue(stats.l0_hits), 0)}</td>
                                <td className="px-3 py-3 text-right">{formatNumber(toNumericValue(stats.l1_hits), 0)}</td>
                                <td className="px-3 py-3 text-right">{formatNumber(toNumericValue(stats.l2_hits), 0)}</td>
                                <td className="px-3 py-3 text-right">{formatNumber(toNumericValue(stats.misses), 0)}</td>
                                <td className="px-3 py-3 text-right">{formatNumber(toNumericValue(stats.writes), 0)}</td>
                                <td className="px-3 py-3 text-right">
                                  {formatMetricValue('cache_hit_rate', toNumericValue(stats.hit_rate))}
                                </td>
                              </tr>
                            ))
                          ) : (
                            <tr>
                              <td colSpan={9} className="px-3 py-8 text-center text-sm text-gray-500">
                                当前还没有缓存统计，先运行一次查询或评测后这里会显示命中数据。
                              </td>
                            </tr>
                          )}
                        </tbody>
                      </table>
                    </div>
                  </div>
                </div>
              </div>
              ) : null}
            </div>

            {evaluationSubTab === 'report' ? (
            evaluationLoading && !evaluationReport ? (
              <div className="mt-6">
                <LoadingSection title="量化评测概览" icon={<Target className="h-5 w-5 text-violet-500" />} />
              </div>
            ) : evaluationError && !evaluationReport ? (
              <div className="mt-6">
                <SectionMessage
                  title="量化评测加载失败"
                  message={evaluationError}
                  icon={<ShieldAlert className="h-6 w-6" />}
                />
              </div>
            ) : !evaluationReport ? (
              <div className="mt-6">
                <SectionMessage
                  title="暂无量化评测报告"
                  message="当前还没有检测到 reports/eval/unified_rag_eval_*.json。生成报告后，这里会自动展示 RAGAS、Recall、MRR 和时延指标。"
                  icon={<FileText className="h-6 w-6" />}
                />
              </div>
            ) : (
              <div className="mt-6 space-y-6">
                {evaluationError ? (
                  <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-700 dark:border-amber-900/40 dark:bg-amber-950/30 dark:text-amber-300">
                    {evaluationError}
                  </div>
                ) : null}

                <section className="rounded-2xl border border-gray-200 bg-white p-4 shadow-sm dark:border-gray-800 dark:bg-gray-900/60">
                  <div className="mb-4 flex items-center justify-between gap-3">
                    <div className="flex items-center gap-2">
                      <Activity className="h-5 w-5 text-violet-500" />
                      <h3 className="text-lg font-semibold">方案对比</h3>
                    </div>
                    <div className="flex flex-wrap items-center justify-end gap-2">
                      <p className="text-xs text-gray-500">点击任一方案，下方所有指标都会切换到对应结果。</p>
                      <button
                        type="button"
                        onClick={handleStartAppendCombo}
                        disabled={appendingEvaluation || isEvaluationBusy || evaluationFeatures.length === 0}
                        className="inline-flex items-center gap-1 rounded-lg border border-violet-200 px-2.5 py-1.5 text-xs font-medium text-violet-700 hover:bg-violet-50 disabled:cursor-not-allowed disabled:opacity-50 dark:border-violet-900 dark:text-violet-300 dark:hover:bg-violet-950/20"
                      >
                        <Plus className="h-3.5 w-3.5" />
                        追加对比项
                      </button>
                    </div>
                  </div>
                  <div className="grid grid-cols-1 gap-3 xl:grid-cols-2">
                    {variantRows.map(([variantName, variant]) => {
                      const isSelected = variantName === activeVariantName;
                      const isFinal = variantName === evaluationReport.final_variant;
                      return (
                        <button
                          key={variantName}
                          type="button"
                          onClick={() => setSelectedVariantName(variantName)}
                          className={`rounded-2xl border p-4 text-left transition-all ${isSelected
                              ? 'border-violet-300 bg-violet-50 shadow-sm dark:border-violet-700 dark:bg-violet-950/20'
                              : 'border-gray-200 bg-white hover:border-violet-200 hover:bg-violet-50/50 dark:border-gray-800 dark:bg-gray-900/40 dark:hover:border-violet-900'
                            }`}
                        >
                          <div className="flex items-start justify-between gap-3">
                            <div className="min-w-0">
                              <div className="flex flex-wrap items-center gap-2">
                                <span className="text-base font-semibold text-gray-900 dark:text-white">
                                  {formatVariantLabel(variantName, variant.technique)}
                                </span>
                                {isSelected ? (
                                  <span className="rounded-full bg-violet-600 px-2 py-0.5 text-[11px] font-medium text-white">
                                    当前查看
                                  </span>
                                ) : null}
                                {isFinal ? (
                                  <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-[11px] font-medium text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300">
                                    最终方案
                                  </span>
                                ) : null}
                              </div>
                              <p className="mt-1 text-sm text-gray-500">{variant.description}</p>
                              {variant.feature_variant?.feature_labels?.length ? (
                                <div className="mt-2 flex flex-wrap gap-1">
                                  {variant.feature_variant.feature_labels.map((label) => (
                                    <span key={label} className="rounded-full bg-white px-2 py-0.5 text-[11px] text-violet-700 dark:bg-gray-900 dark:text-violet-300">
                                      {label}
                                    </span>
                                  ))}
                                </div>
                              ) : null}
                            </div>
                            <div className="rounded-xl bg-violet-100 p-2 text-violet-600 dark:bg-violet-500/10 dark:text-violet-400">
                              <Target className="h-4 w-4" />
                            </div>
                          </div>
                          <div className="mt-4 flex flex-wrap gap-x-4 gap-y-2 text-xs text-gray-500 dark:text-gray-400">
                            {[
                              'factual_correctness',
                              'faithfulness',
                              'recall@5',
                              'mrr@5',
                              'avg_total_duration_ms',
                            ].map((metricKey) => (
                              <div key={metricKey} className="inline-flex items-center gap-1">
                                <span>{formatMetricLabel(metricKey)}</span>
                                <span className="font-semibold text-gray-900 dark:text-white">
                                  {formatMetricValue(metricKey, getSummaryMetric(variant.summary, metricKey))}
                                </span>
                              </div>
                              ))}
                          </div>
                        </button>
                      );
                    })}
                  </div>
                  {appendEditingComboId ? (
                    <div className="mt-4 border-t border-gray-100 pt-4 dark:border-gray-800">
                      <div className="mb-3 flex items-center justify-between gap-2">
                        <div>
                          <h4 className="text-sm font-semibold text-gray-900 dark:text-white">追加功能组合</h4>
                          <p className="mt-1 text-xs text-gray-500">
                            将只运行新增组合，并生成包含当前报告所有方案的新版报告。
                          </p>
                        </div>
                        <button
                          type="button"
                          onClick={() => setAppendEditingComboId(null)}
                          className="rounded-lg p-1.5 text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-800"
                          title="关闭"
                        >
                          <X className="h-4 w-4" />
                        </button>
                      </div>
                      <input
                        type="text"
                        value={appendComboLabel}
                        onChange={(event) => setAppendComboLabel(event.target.value)}
                        placeholder={appendFeatureKeys.length ? formatFeatureComboLabel(appendFeatureKeys, evaluationFeatures) : '选择增强功能后自动生成名称'}
                        className="mb-3 w-full rounded-xl border border-gray-300 bg-white px-3 py-2.5 text-sm dark:border-gray-700 dark:bg-gray-900"
                      />
                      <div className="max-h-[360px] space-y-3 overflow-y-auto pr-1">
                        {groupedEvaluationFeatures.map((group) => (
                          <div key={group.category}>
                            <div className="mb-2 text-xs font-semibold text-gray-500">{group.label}</div>
                            <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
                              {group.features.map((feature) => {
                                const checked = appendFeatureKeys.includes(feature.key);
                                const autoEnabled = autoAppendFeatures.includes(feature.key);
                                return (
                                  <label
                                    key={feature.key}
                                    className={`flex cursor-pointer items-start gap-3 rounded-xl border px-3 py-2.5 transition-colors ${checked || autoEnabled
                                        ? 'border-violet-300 bg-violet-50 dark:border-violet-700 dark:bg-violet-950/20'
                                        : 'border-gray-200 bg-white dark:border-gray-800 dark:bg-gray-950/40'
                                      }`}
                                  >
                                    <input
                                      type="checkbox"
                                      checked={checked}
                                      onChange={() => handleToggleAppendFeature(feature.key)}
                                      className="mt-1 h-4 w-4 rounded border-gray-300 text-violet-600"
                                    />
                                    <div className="min-w-0">
                                      <div className="flex flex-wrap items-center gap-2 text-sm font-medium text-gray-900 dark:text-white">
                                        {feature.label}
                                        {autoEnabled ? (
                                          <span className="rounded-full bg-violet-100 px-2 py-0.5 text-[11px] text-violet-700 dark:bg-violet-950 dark:text-violet-300">
                                            自动启用
                                          </span>
                                        ) : null}
                                      </div>
                                      <div className="mt-1 text-xs text-gray-500">{feature.description}</div>
                                    </div>
                                  </label>
                                );
                              })}
                            </div>
                          </div>
                        ))}
                      </div>
                      <div className="mt-3 flex flex-wrap items-center justify-between gap-2">
                        <div className="text-xs text-gray-500">
                          当前追加：{formatFeatureComboLabel(resolvedAppendFeatures, evaluationFeatures)}
                        </div>
                        <button
                          type="button"
                          onClick={() => void handleAppendEvaluation()}
                          disabled={appendingEvaluation || resolvedAppendFeatures.length === 0}
                          className="inline-flex items-center justify-center gap-2 rounded-xl bg-violet-600 px-3 py-2 text-sm font-medium text-white hover:bg-violet-700 disabled:cursor-not-allowed disabled:opacity-60"
                        >
                          {appendingEvaluation ? <Loader2 className="h-4 w-4 animate-spin" /> : <Plus className="h-4 w-4" />}
                          开始追加测评
                        </button>
                      </div>
                    </div>
                  ) : null}
                </section>

                <section className="rounded-2xl border border-gray-200 bg-white p-4 shadow-sm dark:border-gray-800 dark:bg-gray-900/60">
                  <div className="mb-4 flex flex-col gap-3 lg:flex-row lg:items-start lg:justify-between">
                    <div>
                      <div className="flex items-center gap-2">
                        <Sparkles className="h-5 w-5 text-violet-500" />
                        <h3 className="text-lg font-semibold">指标明细</h3>
                      </div>
                      <p className="mt-1 text-sm text-gray-500">
                        当前查看 {selectedVariantCompactLabel} · 样本 {evaluationReport.case_count} · {evaluationMeta?.dataset_name || '未标注数据集名称'}
                      </p>
                      <p className="mt-1 text-xs text-gray-500">
                        最终方案 {finalVariantCompactLabel}
                        {evaluationMeta?.generated_at
                          ? ` · 生成于 ${evaluationMeta.generated_at.replace('T', ' ').slice(0, 19)}`
                          : ''}
                      </p>
                    </div>
                    <div className="inline-flex flex-wrap gap-1 rounded-xl border border-gray-200 bg-gray-50 p-1 dark:border-gray-800 dark:bg-gray-950/50">
                      {EVALUATION_METRIC_GROUPS.map((group) => {
                        const active = group.key === activeMetricGroup.key;
                        return (
                          <button
                            key={group.key}
                            type="button"
                            onClick={() => setActiveMetricGroupKey(group.key)}
                            className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${active
                                ? 'bg-white text-violet-700 shadow-sm dark:bg-gray-900 dark:text-violet-300'
                                : 'text-gray-600 hover:bg-white/70 hover:text-gray-900 dark:text-gray-400 dark:hover:bg-gray-900/70 dark:hover:text-gray-100'
                              }`}
                          >
                            {group.label}
                          </button>
                        );
                      })}
                    </div>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-sm">
                      <thead className="border-b border-gray-200 text-gray-500 dark:border-gray-800 dark:text-gray-400">
                        <tr>
                          <th className="px-3 py-2 font-medium">指标</th>
                          <th className="px-3 py-2 text-right font-medium">当前值</th>
                          <th className="px-3 py-2 font-medium">覆盖/备注</th>
                        </tr>
                      </thead>
                      <tbody>
                        {activeMetricGroup.metricKeys.map((metricKey) => (
                          <tr key={metricKey} className="border-b border-gray-100 dark:border-gray-800/70">
                            <td className="px-3 py-3 font-medium text-gray-900 dark:text-white">
                              {formatMetricLabel(metricKey)}
                            </td>
                            <td className="px-3 py-3 text-right font-semibold text-gray-900 dark:text-white">
                              {formatMetricValue(metricKey, getSummaryMetric(selectedSummary, metricKey))}
                            </td>
                            <td className="px-3 py-3 text-gray-500">
                              {metricKey === 'cache_hit_rate'
                                ? '来自评测样本级 cache summary 汇总'
                                : formatCoverageLabel(selectedSummary, metricKey) || '-'}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                  {activeMetricGroup.key === 'latency' && selectedSummary?.performance_metrics?.stages?.length ? (
                    <div className="mt-4 border-t border-gray-100 pt-4 dark:border-gray-800">
                      <div className="mb-2 text-sm font-semibold text-gray-900 dark:text-white">阶段耗时</div>
                      <div className="overflow-x-auto">
                        <table className="w-full text-left text-sm">
                          <thead className="border-b border-gray-200 text-gray-500 dark:border-gray-800 dark:text-gray-400">
                            <tr>
                              <th className="px-3 py-2 font-medium">阶段</th>
                              <th className="px-3 py-2 text-right font-medium">平均耗时</th>
                              <th className="px-3 py-2 text-right font-medium">P50 耗时</th>
                              <th className="px-3 py-2 text-right font-medium">P95 耗时</th>
                              <th className="px-3 py-2 text-right font-medium">样本数</th>
                            </tr>
                          </thead>
                          <tbody>
                            {selectedSummary.performance_metrics.stages.map((stage) => (
                              <tr key={stage.stage} className="border-b border-gray-100 dark:border-gray-800/70">
                                <td className="px-3 py-3 font-medium text-gray-900 dark:text-white">{formatStageLabel(stage.stage)}</td>
                                <td className="px-3 py-3 text-right">{formatMs(stage.avg_duration_ms)}</td>
                                <td className="px-3 py-3 text-right">{formatMs(stage.p50_duration_ms)}</td>
                                <td className="px-3 py-3 text-right">{formatMs(stage.p95_duration_ms)}</td>
                                <td className="px-3 py-3 text-right">{stage.count}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    </div>
                  ) : null}
                </section>

                {isBusinessBenchmarkReport ? (
                  <div className="rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-700 dark:border-emerald-900/40 dark:bg-emerald-950/30 dark:text-emerald-300">
                    人工业务基准集：优先结合 LLM Judge 质量、Recall/MRR、时延和缓存口径判断方案表现。
                  </div>
                ) : null}

                {formatQualityJudgeSummary(selectedSummary?.ragas_metadata as Record<string, unknown> | undefined) ? (
                  <div className="rounded-xl border border-violet-200 bg-violet-50 px-4 py-3 text-sm text-violet-700 dark:border-violet-900/40 dark:bg-violet-950/30 dark:text-violet-300">
                    {formatQualityJudgeSummary(selectedSummary?.ragas_metadata as Record<string, unknown> | undefined)}
                  </div>
                ) : null}

                {(selectedGroundTruthSummary || selectedWarnings.length > 0) ? (
                  <section
                    className={`rounded-2xl border p-4 shadow-sm ${selectedWarnings.length > 0 || (selectedGroundTruthSummary?.unresolved_cases || 0) > 0
                        ? 'border-amber-200 bg-amber-50 dark:border-amber-900/40 dark:bg-amber-950/20'
                        : 'border-gray-200 bg-white dark:border-gray-800 dark:bg-gray-900/60'
                      }`}
                  >
                    <div className="mb-4 flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                      <div className="flex items-center gap-2">
                        <ShieldAlert className="h-5 w-5 text-amber-600 dark:text-amber-400" />
                        <h3 className="text-lg font-semibold text-gray-900 dark:text-white">评测口径与数据对齐</h3>
                      </div>
                      <div className="flex flex-wrap items-center gap-2">
                        {showKnowledgeBaseMigrationButton ? (
                          <button
                            type="button"
                            onClick={() => void handleMigrateKnowledgeBaseChunkIds()}
                            disabled={migratingKnowledgeBase}
                            className="inline-flex items-center justify-center gap-2 rounded-xl border border-violet-300 bg-white px-3 py-2 text-sm font-medium text-violet-700 hover:bg-violet-50 disabled:cursor-not-allowed disabled:opacity-60 dark:border-violet-800 dark:bg-gray-950/40 dark:text-violet-300 dark:hover:bg-violet-950/20"
                          >
                            {migratingKnowledgeBase ? <Loader2 className="h-4 w-4 animate-spin" /> : <Database className="h-4 w-4" />}
                            迁移知识库稳定ID
                          </button>
                        ) : null}
                        {showDatasetSyncButton ? (
                          <button
                            type="button"
                            onClick={() => void handleSyncEvaluationDataset()}
                            disabled={syncingEvaluationDataset || !currentEvaluationDatasetPath}
                            className="inline-flex items-center justify-center gap-2 rounded-xl border border-amber-300 bg-white px-3 py-2 text-sm font-medium text-amber-700 hover:bg-amber-50 disabled:cursor-not-allowed disabled:opacity-60 dark:border-amber-800 dark:bg-gray-950/40 dark:text-amber-300 dark:hover:bg-amber-950/20"
                          >
                            {syncingEvaluationDataset ? <Loader2 className="h-4 w-4 animate-spin" /> : <RefreshCcw className="h-4 w-4" />}
                            同步评测数据
                          </button>
                        ) : null}
                      </div>
                    </div>
                    {selectedGroundTruthSummary ? (
                      <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
                        <div className="rounded-xl bg-white/80 px-4 py-3 dark:bg-gray-950/40">
                          <div className="text-xs text-gray-500">检索金标对齐</div>
                          <div className="mt-2 text-2xl font-semibold text-gray-900 dark:text-white">
                            {selectedGroundTruthSummary.resolved_cases}/{selectedGroundTruthSummary.eligible_cases}
                          </div>
                          <div className="mt-1 text-xs text-gray-500">
                            未对齐 {selectedGroundTruthSummary.unresolved_cases}，历史 chunk 重映射 {selectedGroundTruthSummary.stale_declared_cases}
                            {(selectedGroundTruthSummary.item_name_filter_miss_cases || 0) > 0
                              ? `，item 名称修复 ${selectedGroundTruthSummary.item_name_filter_miss_cases}`
                              : ''}
                          </div>
                        </div>
                        <div className="rounded-xl bg-white/80 px-4 py-3 dark:bg-gray-950/40">
                          <div className="text-xs text-gray-500">金标来源</div>
                          <div className="mt-2 text-sm font-medium text-gray-900 dark:text-white">
                            {formatGroundTruthSourceBreakdown(selectedGroundTruthSummary)}
                          </div>
                        </div>
                        <div className="rounded-xl bg-white/80 px-4 py-3 dark:bg-gray-950/40">
                          <div className="text-xs text-gray-500">未对齐原因</div>
                          <div className="mt-2 text-sm font-medium text-gray-900 dark:text-white">
                            {selectedGroundTruthSummary.unresolved_cases > 0
                              ? formatBreakdown(
                                selectedGroundTruthSummary.unresolved_reasons,
                                GROUND_TRUTH_REASON_LABELS,
                              )
                              : '当前方案的检索金标已全部完成对齐'}
                          </div>
                        </div>
                      </div>
                    ) : null}
                    {selectedWarnings.length > 0 ? (
                      <div className="mt-4 space-y-2 text-sm text-amber-800 dark:text-amber-200">
                        {selectedWarnings.map((message) => (
                          <div key={message} className="rounded-xl bg-white/80 px-4 py-3 dark:bg-gray-950/40">
                            {message}
                          </div>
                        ))}
                      </div>
                    ) : null}
                    {activeDatasetSyncResult ? (
                      <div className="mt-4 rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-sm text-emerald-700 dark:border-emerald-900/40 dark:bg-emerald-950/30 dark:text-emerald-300">
                        {activeDatasetSyncResult.message}
                      </div>
                    ) : null}
                    {chunkMigrationResult ? (
                      <div className="mt-4 rounded-xl border border-violet-200 bg-violet-50 px-4 py-3 text-sm text-violet-700 dark:border-violet-900/40 dark:bg-violet-950/30 dark:text-violet-300">
                        {chunkMigrationResult.message} 已同步图谱 {chunkMigrationResult.graph_synced_items}/{chunkMigrationResult.items_scanned} 个 item_name。
                      </div>
                    ) : null}
                  </section>
                ) : null}

                {selectedComparisonRows.length > 0 ? (
                  <section className="rounded-2xl border border-gray-200 bg-white p-4 shadow-sm dark:border-gray-800 dark:bg-gray-900/60">
                    <div className="mb-4 flex items-center gap-2">
                      <Sparkles className="h-5 w-5 text-violet-500" />
                      <h3 className="text-lg font-semibold">增量对比</h3>
                    </div>
                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-sm">
                        <thead className="border-b border-gray-200 text-gray-500 dark:border-gray-800 dark:text-gray-400">
                          <tr>
                            <th className="px-3 py-2 font-medium">对比项</th>
                            {VARIANT_METRIC_KEYS.map((metricKey) => (
                              <th key={metricKey} className="px-3 py-2 text-right font-medium">
                                {formatMetricLabel(metricKey)}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {selectedComparisonRows.map(([comparisonName, comparison]) => (
                            <tr key={comparisonName} className="border-b border-gray-100 dark:border-gray-800/70">
                              <td className="px-3 py-3">
                                <div className="font-medium text-gray-900 dark:text-white">
                                  {formatVariantLabel(
                                    comparison.variant,
                                    evaluationReport.variants?.[comparison.variant]?.technique,
                                  )}{' '}
                                  vs{' '}
                                  {formatVariantLabel(
                                    comparison.compare_to,
                                    evaluationReport.variants?.[comparison.compare_to]?.technique,
                                  )}
                                </div>
                                <div className="mt-1 text-xs text-gray-500">{comparison.technique}</div>
                              </td>
                              {VARIANT_METRIC_KEYS.map((metricKey) => (
                                <td key={metricKey} className="px-3 py-3 text-right">
                                  {formatDelta(metricKey, comparison.overall[metricKey])}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </section>
                ) : null}

                {selectedVariantQueryTypes.length > 0 ? (
                  <section className="rounded-2xl border border-gray-200 bg-white p-4 shadow-sm dark:border-gray-800 dark:bg-gray-900/60">
                    <div className="mb-4 flex items-center gap-2">
                      <Search className="h-5 w-5 text-violet-500" />
                      <h3 className="text-lg font-semibold">按问题类型拆解</h3>
                    </div>
                    <div className="overflow-x-auto">
                      <table className="w-full text-left text-sm">
                        <thead className="border-b border-gray-200 text-gray-500 dark:border-gray-800 dark:text-gray-400">
                          <tr>
                            <th className="px-3 py-2 font-medium">问题类型</th>
                            <th className="px-3 py-2 text-right font-medium">样本数</th>
                            {VARIANT_METRIC_KEYS.map((metricKey) => (
                              <th key={metricKey} className="px-3 py-2 text-right font-medium">
                                {formatMetricLabel(metricKey)}
                              </th>
                            ))}
                          </tr>
                        </thead>
                        <tbody>
                          {selectedVariantQueryTypes.map(([queryType, querySummary]) => (
                            <tr key={queryType} className="border-b border-gray-100 dark:border-gray-800/70">
                              <td className="px-3 py-3 font-medium text-gray-900 dark:text-white">
                                {formatQueryTypeLabel(queryType)}
                              </td>
                              <td className="px-3 py-3 text-right">{querySummary.case_count}</td>
                              {VARIANT_METRIC_KEYS.map((metricKey) => (
                                <td key={metricKey} className="px-3 py-3 text-right">
                                  {formatMetricValue(metricKey, getSummaryMetric(querySummary, metricKey))}
                                </td>
                              ))}
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </section>
                ) : null}

                {selectedRagasErrors.length > 0 ? (
                  <section className="rounded-2xl border border-amber-200 bg-amber-50 p-4 shadow-sm dark:border-amber-900/40 dark:bg-amber-950/20">
                    <div className="mb-3 flex items-center gap-2">
                      <ShieldAlert className="h-5 w-5 text-amber-600 dark:text-amber-400" />
                      <h3 className="text-lg font-semibold text-amber-900 dark:text-amber-100">RAGAS 指标提示</h3>
                    </div>
                    <div className="space-y-2 text-sm text-amber-800 dark:text-amber-200">
                      {selectedRagasErrors.map(([metricKey, message]) => (
                        <div key={metricKey}>
                          <span className="font-medium">{formatMetricLabel(metricKey)}：</span>
                          <span>{message}</span>
                        </div>
                      ))}
                    </div>
                  </section>
                ) : null}
              </div>
            )
            ) : null}
          </section>
        ) : null}
      </div>
    </div>
  );
}
