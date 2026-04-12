import { useCallback, useEffect, useMemo, useState } from 'react';
import {
  Activity,
  Clock3,
  Loader2,
  RefreshCcw,
  Search,
  ShieldAlert,
  Sparkles,
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
import { getPerformanceSummaryData, getPerformanceTimeSeries, getStageBreakdown } from '../services/api';
import type { PerformanceSummary, PerformanceTimePoint, StageBreakdown } from '../types';

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

function formatStageLabel(stage: string) {
  return STAGE_LABELS[stage] || stage.replaceAll('_', ' ');
}

function formatMs(value: number | null | undefined) {
  if (value == null) return '-';
  if (value >= 1000) return `${(value / 1000).toFixed(2)}s`;
  return `${Math.round(value)}ms`;
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

export default function PerformancePage() {
  const [granularity, setGranularity] = useState<'day' | 'hour'>('day');
  const [startDate, setStartDate] = useState('');
  const [endDate, setEndDate] = useState('');

  const [summary, setSummary] = useState<PerformanceSummary | null>(null);
  const [timeSeries, setTimeSeries] = useState<PerformanceTimePoint[]>([]);
  const [stages, setStages] = useState<StageBreakdown[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const loadData = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [summaryData, timeSeriesData, stageData] = await Promise.all([
        getPerformanceSummaryData(startDate || undefined, endDate || undefined),
        getPerformanceTimeSeries(granularity, startDate || undefined, endDate || undefined),
        getStageBreakdown(startDate || undefined, endDate || undefined),
      ]);
      setSummary(summaryData);
      setTimeSeries(timeSeriesData);
      setStages(stageData);
    } catch (err: any) {
      setError(err.message || '加载性能数据失败');
    } finally {
      setLoading(false);
    }
  }, [granularity, startDate, endDate]);

  useEffect(() => {
    loadData();
  }, [loadData]);

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
      stages.map((s) => ({
        ...s,
        stage_label: formatStageLabel(s.stage),
      })),
    [stages],
  );

  if (error && !summary) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3">
        <ShieldAlert className="h-10 w-10 text-red-500" />
        <p className="text-sm text-gray-600 dark:text-gray-300">{error}</p>
        <button
          onClick={loadData}
          className="rounded-lg bg-violet-500 px-4 py-2 text-sm font-medium text-white hover:bg-violet-600"
        >
          重试
        </button>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto p-4 md:p-6">
      <div className="mx-auto flex max-w-6xl flex-col gap-6">
        {/* Header */}
        <div className="flex flex-col gap-4 md:flex-row md:items-center md:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-violet-500">分析看板</p>
            <h1 className="mt-1 flex items-center gap-2 text-2xl font-bold text-gray-900 dark:text-white">
              <Activity className="h-6 w-6 text-violet-500" />
              性能分析
            </h1>
            <p className="text-sm text-gray-500">对比端到端耗时和阶段分布</p>
          </div>
          <div className="flex flex-wrap items-center gap-2">
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
            <button
              onClick={loadData}
              className="rounded-lg border border-gray-200 p-2 text-gray-600 hover:bg-gray-100 dark:border-gray-700 dark:text-gray-300 dark:hover:bg-gray-800"
              title="刷新"
            >
              <RefreshCcw className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Summary Cards */}
        <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
          <StatCard
            title="运行次数"
            value={`${summary?.run_count || 0}`}
            icon={<Activity className="h-5 w-5" />}
            loading={loading && !summary}
          />
          <StatCard
            title="平均总耗时"
            value={formatMs(summary?.avg_total_duration_ms)}
            icon={<Clock3 className="h-5 w-5" />}
            loading={loading && !summary}
          />
          <StatCard
            title="P50 总耗时"
            value={formatMs(summary?.p50_total_duration_ms)}
            icon={<Search className="h-5 w-5" />}
            loading={loading && !summary}
          />
          <StatCard
            title="P95 总耗时"
            value={formatMs(summary?.p95_total_duration_ms)}
            subtitle={summary?.avg_first_answer_ms != null ? `首次回答: ${formatMs(summary.avg_first_answer_ms)}` : undefined}
            icon={<Sparkles className="h-5 w-5" />}
            loading={loading && !summary}
          />
        </div>

        {/* Trend Chart */}
        {loading && trendData.length === 0 ? (
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

        {/* Stage Breakdown Chart */}
        {loading && stageChartData.length === 0 ? (
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

        {/* Stage Table */}
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
                {loading && stages.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="px-3 py-8 text-center text-sm text-gray-500">
                      正在加载阶段明细...
                    </td>
                  </tr>
                ) : (
                  stages.map((stage) => (
                    <tr key={stage.stage} className="border-b border-gray-100 dark:border-gray-800/70">
                      <td className="px-3 py-3 font-medium text-gray-900 dark:text-white">
                        {formatStageLabel(stage.stage)}
                      </td>
                      <td className="px-3 py-3 text-right">{stage.count}</td>
                      <td className="px-3 py-3 text-right">{formatMs(stage.avg_duration_ms)}</td>
                      <td className="px-3 py-3 text-right">{formatMs(stage.p95_duration_ms)}</td>
                      <td className="px-3 py-3 text-right">
                        {(stage.error_rate * 100).toFixed(1)}%
                      </td>
                    </tr>
                  ))
                )}
              </tbody>
            </table>
          </div>
        </section>
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
      <div className="flex h-[320px] items-center justify-center text-sm text-gray-500">
        <Loader2 className="mr-2 h-5 w-5 animate-spin text-violet-500" />
        正在加载图表数据...
      </div>
    </section>
  );
}
