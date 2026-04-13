import { useEffect, useMemo, useState } from 'react';
import { BookOpen, CheckCircle2, Clock3, FileText, RefreshCw } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { getImportList } from '../services/api';
import type { ImportTask } from '../types';

function formatTime(ts?: number) {
  if (!ts) return '未知时间';
  return new Date(ts * 1000).toLocaleString('zh-CN', {
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function getDisplayName(task: ImportTask) {
  if (task.item_name && task.item_name.trim()) return task.item_name.trim();
  if (task.file_title && task.file_title.trim()) return task.file_title.trim();
  return task.file_name.replace(/\.[^.]+$/, '');
}

export default function KnowledgePage() {
  const navigate = useNavigate();
  const [tasks, setTasks] = useState<ImportTask[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [loadError, setLoadError] = useState('');

  useEffect(() => {
    let cancelled = false;

    getImportList(200)
      .then((data) => {
        if (cancelled) return;
        setTasks(data.tasks || []);
        setLoadError('');
      })
      .catch((err: Error) => {
        if (cancelled) return;
        setLoadError(err.message || '知识库记录加载失败');
      })
      .finally(() => {
        if (!cancelled) {
          setIsLoading(false);
        }
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const completedTasks = useMemo(
    () => tasks.filter((task) => task.status === 'completed'),
    [tasks],
  );

  const processingTasks = useMemo(
    () => tasks.filter((task) => task.status === 'processing' || task.status === 'uploading'),
    [tasks],
  );

  const failedTasks = useMemo(
    () => tasks.filter((task) => task.status === 'failed'),
    [tasks],
  );

  return (
    <div className="flex-1 overflow-y-auto p-4 md:p-6">
      <div className="mx-auto max-w-5xl flex flex-col gap-6">
        <div className="flex flex-col gap-4 md:flex-row md:items-end md:justify-between">
          <div>
            <p className="text-xs font-semibold uppercase tracking-[0.2em] text-violet-500">持久化知识库</p>
            <h1 className="mt-1 flex items-center gap-2 text-2xl font-bold text-gray-900 dark:text-white">
              <BookOpen className="h-6 w-6 text-violet-500" />
              已导入内容
            </h1>
            <p className="text-sm text-gray-500">这里展示的是已经落到后端持久化存储中的导入结果，重启后仍可恢复查看。</p>
          </div>
          <div className="flex gap-3">
            <button
              onClick={() => navigate('/import')}
              className="inline-flex items-center gap-2 rounded-xl border border-gray-200 dark:border-gray-700 px-4 py-2 text-sm text-gray-700 dark:text-gray-200 hover:bg-gray-50 dark:hover:bg-gray-800 transition-colors"
            >
              <FileText className="h-4 w-4" />
              继续导入
            </button>
            <button
              onClick={() => navigate('/chat')}
              className="inline-flex items-center gap-2 rounded-xl bg-violet-500 px-4 py-2 text-sm text-white hover:bg-violet-600 transition-colors"
            >
              <BookOpen className="h-4 w-4" />
              去提问
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
          <SummaryCard title="已完成导入" value={completedTasks.length} tone="green" />
          <SummaryCard title="处理中" value={processingTasks.length} tone="violet" />
          <SummaryCard title="失败记录" value={failedTasks.length} tone="red" />
        </div>

        {loadError && (
          <div className="rounded-2xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-600 dark:border-red-900/40 dark:bg-red-900/20 dark:text-red-300">
            {loadError}
          </div>
        )}

        {isLoading ? (
          <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900/60 p-10 text-sm text-gray-500 dark:text-gray-400 flex items-center justify-center gap-3">
            <RefreshCw className="h-4 w-4 animate-spin" />
            正在加载知识库记录...
          </div>
        ) : completedTasks.length === 0 ? (
          <div className="rounded-2xl border border-dashed border-gray-300 dark:border-gray-700 bg-white/80 dark:bg-gray-900/40 p-10 text-center">
            <div className="mx-auto mb-4 flex h-14 w-14 items-center justify-center rounded-2xl bg-violet-50 text-violet-500 dark:bg-violet-900/20">
              <BookOpen className="h-7 w-7" />
            </div>
            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">还没有可恢复的已导入内容</h2>
            <p className="mt-2 text-sm text-gray-500">先去导入产品手册，完成后这里会显示持久化的知识库记录。</p>
            <button
              onClick={() => navigate('/import')}
              className="mt-5 inline-flex items-center gap-2 rounded-xl bg-violet-500 px-4 py-2 text-sm text-white hover:bg-violet-600 transition-colors"
            >
              <FileText className="h-4 w-4" />
              去导入文件
            </button>
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
            {completedTasks.map((task) => (
              <KnowledgeCard key={task.task_id} task={task} />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function SummaryCard({ title, value, tone }: { title: string; value: number; tone: 'green' | 'violet' | 'red' }) {
  const toneClass = {
    green: 'text-green-600 bg-green-50 dark:bg-green-900/20 dark:text-green-300',
    violet: 'text-violet-600 bg-violet-50 dark:bg-violet-900/20 dark:text-violet-300',
    red: 'text-red-600 bg-red-50 dark:bg-red-900/20 dark:text-red-300',
  }[tone];

  return (
    <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900/60 p-5 shadow-sm">
      <div className={`inline-flex rounded-lg px-2.5 py-1 text-xs font-medium ${toneClass}`}>{title}</div>
      <div className="mt-3 text-3xl font-bold text-gray-900 dark:text-white">{value}</div>
    </div>
  );
}

function KnowledgeCard({ task }: { task: ImportTask }) {
  const displayName = getDisplayName(task);

  return (
    <div className="rounded-2xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900/60 p-5 shadow-sm">
      <div className="flex items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="inline-flex items-center gap-1.5 rounded-lg bg-green-50 px-2.5 py-1 text-xs font-medium text-green-600 dark:bg-green-900/20 dark:text-green-300">
            <CheckCircle2 className="h-3.5 w-3.5" />
            已持久化
          </div>
          <h2 className="mt-3 text-lg font-semibold text-gray-900 dark:text-white break-words">{displayName}</h2>
          <p className="mt-1 text-sm text-gray-500 break-all">源文件：{task.file_name}</p>
        </div>
        <div className="inline-flex items-center gap-1.5 text-xs text-gray-500 dark:text-gray-400 shrink-0">
          <Clock3 className="h-3.5 w-3.5" />
          {formatTime(task.updated_at || task.created_at)}
        </div>
      </div>

      {task.done_list.length > 0 && (
        <div className="mt-4 flex flex-wrap gap-2">
          {task.done_list.map((step) => (
            <span
              key={`${task.task_id}-${step}`}
              className="rounded-full bg-gray-100 px-3 py-1 text-xs text-gray-600 dark:bg-gray-800 dark:text-gray-300"
            >
              {step}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
