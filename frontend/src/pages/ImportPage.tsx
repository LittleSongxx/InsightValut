import { useState, useCallback, useRef, useEffect } from 'react';
import { Upload, FileText, CheckCircle2, Loader2, AlertCircle, XCircle, Trash2 } from 'lucide-react';
import { uploadFile, getImportStatus, getImportList, deleteImportTasks } from '../services/api';
import type { ImportTask } from '../types';

interface TaskInfo {
  task_id: string;
  file_name: string;
  file_title?: string;
  item_name?: string;
  status: string;
  done_list: string[];
  running_list: string[];
  uploadProgress: number;
  errorMessage?: string;
}

const ACTIVE_STATUSES = new Set(['uploading', 'processing', 'pending']);

export default function ImportPage() {
  const [tasks, setTasks] = useState<TaskInfo[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [selectedTaskIds, setSelectedTaskIds] = useState<string[]>([]);
  const [isDeleting, setIsDeleting] = useState(false);
  const [deleteFeedback, setDeleteFeedback] = useState<{ tone: 'success' | 'error'; text: string } | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const pollingRef = useRef<Map<string, ReturnType<typeof setInterval>>>(new Map());

  // Load historical tasks on mount + cleanup polling on unmount
  useEffect(() => {
    getImportList()
      .then((data) => {
        const historicalTasks: TaskInfo[] = (data.tasks || []).map((t: ImportTask) => ({
          task_id: t.task_id,
          file_name: t.file_name,
          file_title: t.file_title,
          item_name: t.item_name,
          status: t.status,
          done_list: t.done_list || [],
          running_list: t.running_list || [],
          uploadProgress: 100,
          errorMessage: t.error_message || '',
        }));
        setTasks((prev) => {
          // Merge: keep current in-progress tasks, append historical ones not already present
          const existingIds = new Set(prev.map((p) => p.task_id));
          const merged = [...prev, ...historicalTasks.filter((h) => !existingIds.has(h.task_id))];
          return merged;
        });
      })
      .catch(() => { });

    const pollingMap = pollingRef.current;
    return () => {
      pollingMap.forEach((timer) => clearInterval(timer));
    };
  }, []);

  const startPolling = useCallback((taskId: string) => {
    const timer = setInterval(async () => {
      try {
        const res = await getImportStatus(taskId);
        setTasks((prev) =>
          prev.map((t) =>
            t.task_id === taskId
              ? {
                ...t,
                status: res.status,
                done_list: res.done_list || [],
                running_list: res.running_list || [],
                errorMessage: res.error_message || t.errorMessage,
                item_name: res.item_name || t.item_name,
                file_title: res.file_title || t.file_title,
              }
              : t,
          ),
        );
        if (res.status === 'completed' || res.status === 'failed') {
          clearInterval(pollingRef.current.get(taskId));
          pollingRef.current.delete(taskId);
        }
      } catch {
        // Ignore polling errors
      }
    }, 2000);
    pollingRef.current.set(taskId, timer);
  }, []);

  const handleUpload = useCallback(
    async (file: File) => {
      const tempId = `temp-${Date.now()}`;
      const newTask: TaskInfo = {
        task_id: tempId,
        file_name: file.name,
        status: 'uploading',
        done_list: [],
        running_list: [],
        uploadProgress: 0,
      };
      setTasks((prev) => [newTask, ...prev]);

      try {
        const res = await uploadFile(file, (pct) => {
          setTasks((prev) =>
            prev.map((t) => (t.task_id === tempId ? { ...t, uploadProgress: pct } : t)),
          );
        });

        setTasks((prev) =>
          prev.map((t) =>
            t.task_id === tempId
              ? { ...t, task_id: res.task_id, status: 'processing', uploadProgress: 100 }
              : t,
          ),
        );

        startPolling(res.task_id);
      } catch (err: unknown) {
        const message = err instanceof Error && err.message ? err.message : '上传失败';
        setTasks((prev) =>
          prev.map((t) =>
            t.task_id === tempId
              ? { ...t, status: 'failed', errorMessage: message }
              : t,
          ),
        );
      }
    },
    [startPolling],
  );

  const handleFiles = useCallback(
    (files: FileList) => {
      Array.from(files).forEach((f) => {
        if (f.name.endsWith('.pdf') || f.name.endsWith('.md') || f.name.endsWith('.txt')) {
          handleUpload(f);
        }
      });
    },
    [handleUpload],
  );

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setIsDragging(false);
      if (e.dataTransfer.files.length > 0) {
        handleFiles(e.dataTransfer.files);
      }
    },
    [handleFiles],
  );

  const deletableTaskIds = tasks
    .filter((task) => !ACTIVE_STATUSES.has(task.status) && !task.task_id.startsWith('temp-'))
    .map((task) => task.task_id);

  const allDeletableSelected = deletableTaskIds.length > 0 && deletableTaskIds.every((taskId) => selectedTaskIds.includes(taskId));

  const toggleTaskSelection = useCallback((taskId: string) => {
    setSelectedTaskIds((prev) => (
      prev.includes(taskId)
        ? prev.filter((id) => id !== taskId)
        : [...prev, taskId]
    ));
  }, []);

  const toggleSelectAll = useCallback(() => {
    setSelectedTaskIds((prev) => (allDeletableSelected ? [] : deletableTaskIds));
  }, [allDeletableSelected, deletableTaskIds]);

  const handleDeleteSelected = useCallback(async () => {
    if (selectedTaskIds.length === 0 || isDeleting) {
      return;
    }

    const selectedTasks = tasks.filter((task) => selectedTaskIds.includes(task.task_id));
    const summary = selectedTasks
      .slice(0, 5)
      .map((task) => `- ${task.item_name || task.file_title || task.file_name}`)
      .join('\n');
    const suffix = selectedTasks.length > 5 ? `\n- 以及另外 ${selectedTasks.length - 5} 条记录` : '';
    const confirmed = window.confirm(`确定要真正删除选中的 ${selectedTasks.length} 条导入记录吗？\n\n删除后将同步移除对应的持久化记录，且无法恢复。\n\n${summary}${suffix}`);

    if (!confirmed) {
      return;
    }

    setIsDeleting(true);
    setDeleteFeedback(null);

    try {
      const result = await deleteImportTasks(selectedTaskIds);
      const deletedIds = new Set(result.deleted_task_ids || []);

      deletedIds.forEach((taskId) => {
        const timer = pollingRef.current.get(taskId);
        if (timer) {
          clearInterval(timer);
          pollingRef.current.delete(taskId);
        }
      });

      setTasks((prev) => prev.filter((task) => !deletedIds.has(task.task_id)));
      setSelectedTaskIds((prev) => prev.filter((taskId) => !deletedIds.has(taskId)));

      if (result.skipped.length > 0) {
        const reasons = result.skipped.map((item) => `${item.task_id}: ${item.reason}`).join('；');
        setDeleteFeedback({
          tone: result.deleted_count > 0 ? 'success' : 'error',
          text: `已删除 ${result.deleted_count} 条记录。未删除项：${reasons}`,
        });
      } else {
        setDeleteFeedback({ tone: 'success', text: `已删除 ${result.deleted_count} 条导入记录。` });
      }
    } catch (err) {
      const message = err instanceof Error && err.message ? err.message : '批量删除失败';
      setDeleteFeedback({ tone: 'error', text: message });
    } finally {
      setIsDeleting(false);
    }
  }, [isDeleting, selectedTaskIds, tasks]);

  return (
    <div className="flex-1 overflow-y-auto p-4 md:p-6">
      <div className="mx-auto max-w-4xl flex flex-col gap-6">
        {/* Header */}
        <div>
          <p className="text-xs font-semibold uppercase tracking-[0.2em] text-violet-500">知识管理</p>
          <h1 className="mt-1 flex items-center gap-2 text-2xl font-bold text-gray-900 dark:text-white">
            <Upload className="h-6 w-6 text-violet-500" />
            文件导入
          </h1>
          <p className="text-sm text-gray-500">上传产品手册（PDF / Markdown / TXT），自动解析并导入产品知识库</p>
        </div>

        {/* Drop Zone */}
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setIsDragging(true);
          }}
          onDragLeave={() => setIsDragging(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className={`
            relative cursor-pointer rounded-2xl border-2 border-dashed p-12 text-center transition-all duration-300
            ${isDragging
              ? 'border-violet-400 bg-violet-50 dark:bg-violet-900/20'
              : 'border-gray-300 dark:border-gray-700 hover:border-violet-400 hover:bg-violet-50/50 dark:hover:bg-violet-900/10'
            }
          `}
        >
          <input
            ref={fileInputRef}
            type="file"
            multiple
            accept=".pdf,.md,.txt"
            className="hidden"
            onChange={(e) => e.target.files && handleFiles(e.target.files)}
          />
          <Upload
            className={`mx-auto h-12 w-12 mb-4 transition-colors ${isDragging ? 'text-violet-500' : 'text-gray-400'
              }`}
          />
          <p className="text-lg font-medium text-gray-700 dark:text-gray-300">
            拖拽文件到此处，或点击选择
          </p>
          <p className="text-sm text-gray-500 mt-1">支持 PDF、Markdown、TXT 格式</p>
        </div>

        {/* Task List */}
        {tasks.length > 0 && (
          <div className="space-y-3">
            <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
              <div>
                <h2 className="text-lg font-semibold text-gray-900 dark:text-white">导入任务</h2>
                <p className="text-xs text-gray-500">已完成或失败的记录可批量删除，处理中任务会被自动跳过。</p>
              </div>
              <div className="flex flex-col gap-2 sm:flex-row sm:items-center">
                <label className="inline-flex items-center gap-2 text-sm text-gray-600 dark:text-gray-300">
                  <input
                    type="checkbox"
                    className="h-4 w-4 rounded border-gray-300 text-violet-500 focus:ring-violet-500"
                    checked={allDeletableSelected}
                    disabled={deletableTaskIds.length === 0 || isDeleting}
                    onChange={toggleSelectAll}
                  />
                  全选可删除项
                </label>
                <button
                  type="button"
                  disabled={selectedTaskIds.length === 0 || isDeleting}
                  onClick={handleDeleteSelected}
                  className="inline-flex items-center justify-center gap-2 rounded-xl bg-red-500 px-4 py-2 text-sm font-medium text-white transition-colors hover:bg-red-600 disabled:cursor-not-allowed disabled:bg-red-300 dark:disabled:bg-red-900/40"
                >
                  <Trash2 className="h-4 w-4" />
                  {isDeleting ? '删除中...' : `批量删除${selectedTaskIds.length > 0 ? ` (${selectedTaskIds.length})` : ''}`}
                </button>
              </div>
            </div>
            {deleteFeedback && (
              <div className={`rounded-xl px-4 py-3 text-sm ${deleteFeedback.tone === 'success'
                ? 'border border-green-200 bg-green-50 text-green-700 dark:border-green-900/40 dark:bg-green-900/20 dark:text-green-300'
                : 'border border-red-200 bg-red-50 text-red-700 dark:border-red-900/40 dark:bg-red-900/20 dark:text-red-300'
                }`}>
                {deleteFeedback.text}
              </div>
            )}
            {tasks.map((task) => (
              <TaskCard
                key={task.task_id}
                task={task}
                selected={selectedTaskIds.includes(task.task_id)}
                selectable={!ACTIVE_STATUSES.has(task.status) && !task.task_id.startsWith('temp-')}
                onToggleSelect={toggleTaskSelection}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

function TaskCard({
  task,
  selected,
  selectable,
  onToggleSelect,
}: {
  task: TaskInfo;
  selected: boolean;
  selectable: boolean;
  onToggleSelect: (taskId: string) => void;
}) {
  const displayName = (task.item_name || task.file_title || task.file_name.replace(/\.[^.]+$/, '')).trim();
  const statusIcon = {
    uploading: <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />,
    processing: <Loader2 className="w-5 h-5 text-violet-500 animate-spin" />,
    completed: <CheckCircle2 className="w-5 h-5 text-green-500" />,
    failed: <XCircle className="w-5 h-5 text-red-500" />,
    pending: <AlertCircle className="w-5 h-5 text-gray-400" />,
  }[task.status] || <AlertCircle className="w-5 h-5 text-gray-400" />;

  const statusLabel = {
    uploading: '上传中',
    processing: '处理中',
    completed: '已完成',
    failed: '失败',
    pending: '等待中',
  }[task.status] || task.status;

  return (
    <div className={`rounded-2xl border bg-white p-4 shadow-sm transition-colors dark:bg-gray-900/60 ${selected
      ? 'border-violet-400 ring-2 ring-violet-100 dark:border-violet-500 dark:ring-violet-900/40'
      : 'border-gray-200 dark:border-gray-800'
      }`}>
      <div className="flex items-center gap-3 mb-3">
        <input
          type="checkbox"
          className="h-4 w-4 rounded border-gray-300 text-violet-500 focus:ring-violet-500 disabled:cursor-not-allowed disabled:opacity-40"
          checked={selected}
          disabled={!selectable}
          onChange={() => onToggleSelect(task.task_id)}
        />
        <FileText className="w-5 h-5 text-gray-400 shrink-0" />
        <span className="font-medium text-gray-800 dark:text-gray-200 truncate flex-1">
          {task.file_name}
        </span>
        <div className="flex items-center gap-1.5">
          {statusIcon}
          <span className="text-sm text-gray-500">{statusLabel}</span>
        </div>
      </div>

      <div className="mb-3 text-sm text-gray-600 dark:text-gray-300">
        识别内容：{displayName}
      </div>

      {/* Upload progress */}
      {task.status === 'uploading' && (
        <div className="mb-3">
          <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-2">
            <div
              className="bg-blue-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${task.uploadProgress}%` }}
            />
          </div>
          <p className="text-xs text-gray-500 mt-1">{task.uploadProgress}%</p>
        </div>
      )}

      {/* Step progress */}
      {task.status === 'failed' && task.errorMessage && (
        <div className="mb-3 rounded-lg bg-red-50 dark:bg-red-900/20 p-3 text-sm text-red-600 dark:text-red-400">
          {task.errorMessage}
        </div>
      )}

      {(task.done_list.length > 0 || task.running_list.length > 0) && (
        <div className="space-y-1">
          {task.done_list.map((step, i) => (
            <div key={i} className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
              <CheckCircle2 className="w-3.5 h-3.5 text-green-500 shrink-0" />
              <span>{step}</span>
            </div>
          ))}
          {task.running_list.map((step, i) => (
            <div key={i} className="flex items-center gap-2 text-xs text-violet-600 dark:text-violet-400">
              <Loader2 className="w-3.5 h-3.5 animate-spin shrink-0" />
              <span>{step}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
