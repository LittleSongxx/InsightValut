import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Activity,
  BarChart3,
  CheckCircle2,
  Clock3,
  Database,
  FileText,
  Loader2,
  Play,
  RefreshCcw,
  Search,
  ShieldAlert,
  Sparkles,
  Target,
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
  createEvaluationJob,
  getEvaluationConfig,
  getEvaluationJob,
  getEvaluationJobs,
  getEvaluationReport,
  getEvaluationReports,
  getPerformanceSummaryData,
  getPerformanceTimeSeries,
  getStageBreakdown,
} from '../services/api';
import type {
  EvaluationConfig,
  EvaluationJob,
  EvaluationMetricDelta,
  EvaluationReportDetail,
  EvaluationReportListItem,
  EvaluationSummary,
  PerformanceSummary,
  PerformanceTimePoint,
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
  'hit@1': 'Hit@1',
  'hit@3': 'Hit@3',
  'recall@3': 'Recall@3',
  'mrr@3': 'MRR@3',
  avg_total_duration_ms: '平均总耗时',
  p95_total_duration_ms: 'P95 总耗时',
  empty_retrieval_rate: '空检索率',
  empty_answer_rate: '空回答率',
  crag_retry_rate: 'CRAG 重试率',
  hallucination_retry_rate: '幻觉重试率',
  graph_preferred_rate: '图优先比例',
  need_rag_rate: 'RAG 使用率',
  error_rate: '错误率',
};

const HEADLINE_METRIC_KEYS = [
  'factual_correctness',
  'faithfulness',
  'response_relevancy',
  'llm_context_recall',
  'recall@3',
  'mrr@3',
] as const;

const VARIANT_METRIC_KEYS = [
  'factual_correctness',
  'faithfulness',
  'recall@3',
  'mrr@3',
  'avg_total_duration_ms',
] as const;

const FRONTEND_EVALUATION_TEMPLATE_PATH = '/app/docs/graph_eval_cases.docs.json';

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

function formatVariantLabel(variantName: string, technique?: string) {
  if (technique) return technique;
  return variantName
    .split('_')
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(' ');
}

function formatMs(value: number | null | undefined) {
  if (value == null) return '-';
  if (value >= 1000) return `${(value / 1000).toFixed(2)}s`;
  return `${Math.round(value)}ms`;
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
    ].includes(metricKey)
  );
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
    completed: '已完成',
    failed: '失败',
  };
  return mapping[status] || status;
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
        <div className="min-w-0">
          <p className="text-xs uppercase tracking-[0.15em] text-gray-500">{title}</p>
          <p className="mt-2 text-2xl font-bold text-gray-900 dark:text-white">{value}</p>
          {subtitle && <p className="mt-1 text-xs text-gray-500">{subtitle}</p>}
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
  const [selectedVariants, setSelectedVariants] = useState<string[]>([]);
  const [startingEvaluation, setStartingEvaluation] = useState(false);
  const [evaluationActionError, setEvaluationActionError] = useState<string | null>(null);

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
      setEvaluationActionError(null);
      setEvaluationConfig(configData);
      setDatasetPath((prev) => prev || resolveTemplateDatasetPath(configData.template_dataset_path));
      setSelectedVariants((prev) => (prev.length > 0 ? prev : configData.default_variants || []));
      const jobs = jobsData.jobs || [];
      setEvaluationJobs(jobs);
      setActiveEvaluationJob((prev) => {
        if (prev && (prev.status === 'pending' || prev.status === 'running')) {
          const updated = jobs.find((job) => job.job_id === prev.job_id);
          return updated || prev;
        }
        return jobs[0] || null;
      });
    } catch (err: unknown) {
      setEvaluationActionError(err instanceof Error && err.message ? err.message : '加载评测配置失败');
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

  const handleToggleVariant = useCallback((variantName: string) => {
    setSelectedVariants((prev) =>
      prev.includes(variantName)
        ? prev.filter((item) => item !== variantName)
        : [...prev, variantName],
    );
  }, []);

  const handleStartEvaluation = useCallback(async () => {
    const trimmedDatasetPath = datasetPath.trim();
    if (!trimmedDatasetPath) {
      setEvaluationActionError('请先填写评测数据集路径');
      return;
    }
    if (selectedVariants.length === 0) {
      setEvaluationActionError('请至少选择一个评测变体');
      return;
    }

    setStartingEvaluation(true);
    setEvaluationActionError(null);
    try {
      const job = await createEvaluationJob({
        dataset_path: trimmedDatasetPath,
        variants: selectedVariants,
      });
      setActiveEvaluationJob(job);
      setEvaluationJobs((prev) => upsertJob(prev, job));
      void loadEvaluationRuntime();
    } catch (err: unknown) {
      setEvaluationActionError(err instanceof Error && err.message ? err.message : '启动评测失败');
    } finally {
      setStartingEvaluation(false);
    }
  }, [datasetPath, loadEvaluationRuntime, selectedVariants]);

  const handleRefresh = useCallback(() => {
    if (activeTab === 'runtime') {
      void loadPerformanceData();
      return;
    }
    void loadEvaluationData(selectedReportId || undefined);
    void loadEvaluationRuntime();
  }, [activeTab, loadEvaluationData, loadEvaluationRuntime, loadPerformanceData, selectedReportId]);

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
    if (!activeEvaluationJob || !['pending', 'running'].includes(activeEvaluationJob.status)) {
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
          if (latestJob.report_id) {
            void loadEvaluationData(latestJob.report_id);
          } else {
            void loadEvaluationData();
          }
        } else if (latestJob.status === 'failed') {
          window.clearInterval(timer);
          setEvaluationActionError(latestJob.error || '评测任务失败');
          void loadEvaluationRuntime();
        }
      } catch (err: unknown) {
        window.clearInterval(timer);
        setEvaluationActionError(err instanceof Error && err.message ? err.message : '轮询评测任务失败');
      }
    }, 2500);
    return () => window.clearInterval(timer);
  }, [activeEvaluationJob, loadEvaluationData, loadEvaluationRuntime]);

  const trendData = useMemo(
    () =>
      timeSeries.map((point) => ({
        period: new Date(point.period).toLocaleString('zh-CN', {
          month: 'short',
          day: 'numeric',
          ...(granularity === 'hour' ? { hour: '2-digit' as const } : {}),
        }),
        avgDuration: point.avg_total_duration_ms ? Number((point.avg_total_duration_ms / 1000).toFixed(2)) : 0,
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

  const finalSummary = evaluationReport?.final_system_metrics ?? null;
  const finalVariantPayload = evaluationReport?.final_variant
    ? evaluationReport.variants?.[evaluationReport.final_variant]
    : undefined;
  const finalVariantQueryTypes = useMemo(
    () => Object.entries(finalVariantPayload?.by_query_type || {}),
    [finalVariantPayload],
  );
  const variantRows = useMemo(() => Object.entries(evaluationReport?.variants || {}), [evaluationReport]);
  const comparisonRows = useMemo(() => Object.entries(evaluationReport?.comparisons || {}), [evaluationReport]);
  const ragasErrors = useMemo(
    () => Object.entries(finalSummary?.ragas_errors || {}).filter(([, message]) => Boolean(message)),
    [finalSummary],
  );
  const preferredTemplateDatasetPath = useMemo(
    () => resolveTemplateDatasetPath(evaluationConfig?.template_dataset_path),
    [evaluationConfig?.template_dataset_path],
  );
  const isEvaluationBusy =
    evaluationLoading || startingEvaluation || ['pending', 'running'].includes(activeEvaluationJob?.status || '');

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
                      className={`inline-flex items-center gap-2 rounded-xl px-4 py-2 text-sm font-medium transition-colors ${
                        isActive
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
                        {report.generated_at.replace('T', ' ').slice(0, 19)} · {report.dataset_name || report.file_name}
                      </option>
                    ))
                  )}
                </select>
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
            subtitle={summary?.avg_first_answer_ms != null ? `首次回答: ${formatMs(summary.avg_first_answer_ms)}` : undefined}
            icon={<Sparkles className="h-5 w-5" />}
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
                  <th className="px-3 py-2 text-right font-medium">P95</th>
                  <th className="px-3 py-2 text-right font-medium">错误率</th>
                </tr>
              </thead>
              <tbody>
                {performanceLoading && stages.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="px-3 py-8 text-center text-sm text-gray-500">
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
                      <td className="px-3 py-3 text-right">{formatMs(stage.p95_duration_ms)}</td>
                      <td className="px-3 py-3 text-right">{(stage.error_rate * 100).toFixed(1)}%</td>
                    </tr>
                  ))
                ) : (
                  <tr>
                    <td colSpan={5} className="px-3 py-8 text-center text-sm text-gray-500">
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

            <div className="rounded-2xl border border-gray-200 bg-gray-50/70 p-4 dark:border-gray-800 dark:bg-gray-950/40">
              <div className="flex flex-col gap-4">
                <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                  <div>
                    <h3 className="flex items-center gap-2 text-base font-semibold text-gray-900 dark:text-white">
                      <Play className="h-4 w-4 text-violet-500" />
                      运行统一评测
                    </h3>
                    <p className="mt-1 text-sm text-gray-500">
                      填写评测数据集路径，勾选要对比的变体，系统会在后台执行并自动生成报告。
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

                <div className="grid grid-cols-1 gap-4 xl:grid-cols-[1.3fr_1fr]">
                  <div className="space-y-4">
                    <div>
                      <label className="mb-2 block text-sm font-medium text-gray-700 dark:text-gray-300">数据集路径</label>
                      <input
                        type="text"
                        value={datasetPath}
                        onChange={(e) => setDatasetPath(e.target.value)}
                        placeholder={preferredTemplateDatasetPath || '请输入 JSON / JSONL 数据集路径'}
                        className="w-full rounded-xl border border-gray-300 bg-white px-3 py-2.5 text-sm dark:border-gray-700 dark:bg-gray-900"
                      />
                      <p className="mt-2 text-xs text-gray-500">模板：{preferredTemplateDatasetPath}</p>
                    </div>

                    <div>
                      <div className="mb-2 flex items-center justify-between gap-2">
                        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">评测变体</label>
                        <button
                          type="button"
                          onClick={() => setSelectedVariants(evaluationConfig?.default_variants || [])}
                          className="text-xs text-violet-600 hover:text-violet-700 dark:text-violet-400"
                        >
                          恢复默认
                        </button>
                      </div>
                      <div className="grid grid-cols-1 gap-2 md:grid-cols-2">
                        {(evaluationConfig?.variant_catalog || []).map((variant) => {
                          const checked = selectedVariants.includes(variant.name);
                          return (
                            <label
                              key={variant.name}
                              className={`flex cursor-pointer items-start gap-3 rounded-xl border px-3 py-3 transition-colors ${
                                checked
                                  ? 'border-violet-300 bg-violet-50 dark:border-violet-700 dark:bg-violet-950/20'
                                  : 'border-gray-200 bg-white dark:border-gray-800 dark:bg-gray-900'
                              }`}
                            >
                              <input
                                type="checkbox"
                                checked={checked}
                                onChange={() => handleToggleVariant(variant.name)}
                                className="mt-1 h-4 w-4 rounded border-gray-300 text-violet-600"
                              />
                              <div className="min-w-0">
                                <div className="text-sm font-medium text-gray-900 dark:text-white">{variant.technique}</div>
                                <div className="mt-1 text-xs text-gray-500">{variant.description}</div>
                              </div>
                            </label>
                          );
                        })}
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
                          className={`rounded-full px-2.5 py-1 text-xs font-medium ${
                            activeEvaluationJob.status === 'completed'
                              ? 'bg-emerald-100 text-emerald-700 dark:bg-emerald-950/40 dark:text-emerald-300'
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
                          {activeEvaluationJob.report_id ? (
                            <div className="flex items-center justify-between gap-3 text-xs text-gray-500">
                              <span>输出报告</span>
                              <span className="text-gray-700 dark:text-gray-300">{activeEvaluationJob.report_id}</span>
                            </div>
                          ) : null}
                        </>
                      ) : null}
                    </div>

                    <button
                      type="button"
                      onClick={() => void handleStartEvaluation()}
                      disabled={startingEvaluation || activeEvaluationJob?.status === 'running'}
                      className="mt-4 inline-flex w-full items-center justify-center gap-2 rounded-xl bg-violet-600 px-4 py-2.5 text-sm font-medium text-white hover:bg-violet-700 disabled:cursor-not-allowed disabled:opacity-60"
                    >
                      {startingEvaluation || activeEvaluationJob?.status === 'running' ? (
                        <Loader2 className="h-4 w-4 animate-spin" />
                      ) : (
                        <Play className="h-4 w-4" />
                      )}
                      开始评测
                    </button>
                  </div>
                </div>

                {evaluationActionError ? (
                  <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-700 dark:border-amber-900/40 dark:bg-amber-950/30 dark:text-amber-300">
                    {evaluationActionError}
                  </div>
                ) : null}

                {evaluationJobs.length > 0 ? (
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
                            className={`border-b border-gray-100 dark:border-gray-800/70 ${
                              activeEvaluationJob?.job_id === job.job_id ? 'bg-violet-50/70 dark:bg-violet-950/20' : ''
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
              </div>
            </div>
          </div>

          {evaluationLoading && !evaluationReport ? (
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
              <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
                <StatCard
                  title="评测样本"
                  value={`${evaluationReport.case_count}`}
                  subtitle={evaluationMeta?.dataset_name || '未标注数据集名称'}
                  icon={<Database className="h-5 w-5" />}
                  loading={evaluationLoading && !evaluationMeta}
                />
                <StatCard
                  title="最终方案"
                  value={formatVariantLabel(evaluationReport.final_variant, finalVariantPayload?.technique)}
                  subtitle={evaluationMeta?.generated_at ? `生成于 ${evaluationMeta.generated_at.replace('T', ' ').slice(0, 19)}` : undefined}
                  icon={<Target className="h-5 w-5" />}
                  loading={evaluationLoading && !finalSummary}
                />
                {HEADLINE_METRIC_KEYS.map((metricKey) => (
                  <StatCard
                    key={metricKey}
                    title={formatMetricLabel(metricKey)}
                    value={formatMetricValue(metricKey, getSummaryMetric(finalSummary, metricKey))}
                    subtitle={
                      finalSummary?.ragas_coverage?.[metricKey] != null
                        ? `覆盖样本 ${finalSummary.ragas_coverage[metricKey]}`
                        : undefined
                    }
                    icon={<Sparkles className="h-5 w-5" />}
                    loading={evaluationLoading && !finalSummary}
                  />
                ))}
              </div>

              {evaluationError ? (
                <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-700 dark:border-amber-900/40 dark:bg-amber-950/30 dark:text-amber-300">
                  {evaluationError}
                </div>
              ) : null}

              <section className="rounded-2xl border border-gray-200 bg-white p-4 shadow-sm dark:border-gray-800 dark:bg-gray-900/60">
                <div className="mb-4 flex items-center gap-2">
                  <Activity className="h-5 w-5 text-violet-500" />
                  <h3 className="text-lg font-semibold">方案对比</h3>
                </div>
                <div className="overflow-x-auto">
                  <table className="w-full text-left text-sm">
                    <thead className="border-b border-gray-200 text-gray-500 dark:border-gray-800 dark:text-gray-400">
                      <tr>
                        <th className="px-3 py-2 font-medium">方案</th>
                        {VARIANT_METRIC_KEYS.map((metricKey) => (
                          <th key={metricKey} className="px-3 py-2 text-right font-medium">
                            {formatMetricLabel(metricKey)}
                          </th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {variantRows.map(([variantName, variant]) => {
                        const isFinal = variantName === evaluationReport.final_variant;
                        return (
                          <tr
                            key={variantName}
                            className={`border-b border-gray-100 dark:border-gray-800/70 ${
                              isFinal ? 'bg-violet-50/70 dark:bg-violet-950/20' : ''
                            }`}
                          >
                            <td className="px-3 py-3">
                              <div className="font-medium text-gray-900 dark:text-white">
                                {formatVariantLabel(variantName, variant.technique)}
                              </div>
                              <div className="mt-1 text-xs text-gray-500">{variant.description}</div>
                            </td>
                            {VARIANT_METRIC_KEYS.map((metricKey) => (
                              <td key={metricKey} className="px-3 py-3 text-right">
                                {formatMetricValue(metricKey, getSummaryMetric(variant.summary, metricKey))}
                              </td>
                            ))}
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </section>

              {comparisonRows.length > 0 ? (
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
                        {comparisonRows.map(([comparisonName, comparison]) => (
                          <tr key={comparisonName} className="border-b border-gray-100 dark:border-gray-800/70">
                            <td className="px-3 py-3">
                              <div className="font-medium text-gray-900 dark:text-white">
                                {formatVariantLabel(comparison.variant)} vs {formatVariantLabel(comparison.compare_to)}
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

              {finalVariantQueryTypes.length > 0 ? (
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
                        {finalVariantQueryTypes.map(([queryType, querySummary]) => (
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

              {ragasErrors.length > 0 ? (
                <section className="rounded-2xl border border-amber-200 bg-amber-50 p-4 shadow-sm dark:border-amber-900/40 dark:bg-amber-950/20">
                  <div className="mb-3 flex items-center gap-2">
                    <ShieldAlert className="h-5 w-5 text-amber-600 dark:text-amber-400" />
                    <h3 className="text-lg font-semibold text-amber-900 dark:text-amber-100">RAGAS 指标提示</h3>
                  </div>
                  <div className="space-y-2 text-sm text-amber-800 dark:text-amber-200">
                    {ragasErrors.map(([metricKey, message]) => (
                      <div key={metricKey}>
                        <span className="font-medium">{formatMetricLabel(metricKey)}：</span>
                        <span>{message}</span>
                      </div>
                    ))}
                  </div>
                </section>
              ) : null}
            </div>
          )}
        </section>
        ) : null}
      </div>
    </div>
  );
}
