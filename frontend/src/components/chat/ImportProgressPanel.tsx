import { FileText, Loader2, CheckCircle2, XCircle, ChevronDown, ChevronUp } from 'lucide-react';
import { useState } from 'react';
import type { UploadingFile } from './ChatInput';

interface ImportProgressPanelProps {
  files: UploadingFile[];
}

export function ImportProgressPanel({ files }: ImportProgressPanelProps) {
  // Only show files that are actively processing or recently completed/failed
  const activeFiles = files.filter(
    (f) => f.status === 'uploading' || f.status === 'processing' || f.status === 'completed' || f.status === 'failed',
  );
  if (activeFiles.length === 0) return null;

  return (
    <div className="space-y-2">
      {activeFiles.map((file) => (
        <ImportFileCard key={file.id} file={file} />
      ))}
    </div>
  );
}

function ImportFileCard({ file }: { file: UploadingFile }) {
  const [collapsed, setCollapsed] = useState(false);

  const hasDoneSteps = (file.done_list?.length ?? 0) > 0;
  const hasRunningSteps = (file.running_list?.length ?? 0) > 0;
  const hasSteps = hasDoneSteps || hasRunningSteps;

  const statusIcon = {
    uploading: <Loader2 className="w-4 h-4 text-blue-500 animate-spin shrink-0" />,
    processing: <Loader2 className="w-4 h-4 text-violet-500 animate-spin shrink-0" />,
    completed: <CheckCircle2 className="w-4 h-4 text-green-500 shrink-0" />,
    failed: <XCircle className="w-4 h-4 text-red-500 shrink-0" />,
  }[file.status];

  const statusLabel = {
    uploading: '上传中',
    processing: '处理中',
    completed: '已完成',
    failed: '失败',
  }[file.status];

  return (
    <div className="rounded-xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900/60 shadow-sm overflow-hidden">
      {/* Header */}
      <div
        className="flex items-center gap-2.5 px-3 py-2.5 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors"
        onClick={() => setCollapsed(!collapsed)}
      >
        <FileText className="w-4 h-4 text-gray-400 shrink-0" />
        <span className="text-sm font-medium text-gray-800 dark:text-gray-200 truncate flex-1">
          {file.name}
        </span>
        <div className="flex items-center gap-1.5">
          {statusIcon}
          <span className="text-xs text-gray-500">{statusLabel}</span>
        </div>
        {hasSteps && (
          collapsed
            ? <ChevronDown className="w-3.5 h-3.5 text-gray-400 shrink-0" />
            : <ChevronUp className="w-3.5 h-3.5 text-gray-400 shrink-0" />
        )}
      </div>

      {/* Upload progress bar */}
      {file.status === 'uploading' && (
        <div className="px-3 pb-2">
          <div className="w-full bg-gray-200 dark:bg-gray-800 rounded-full h-1.5">
            <div
              className="bg-blue-500 h-1.5 rounded-full transition-all duration-300"
              style={{ width: `${file.progress}%` }}
            />
          </div>
          <p className="text-[10px] text-gray-400 mt-0.5">{file.progress}%</p>
        </div>
      )}

      {/* Error message */}
      {file.status === 'failed' && file.errorMessage && (
        <div className="mx-3 mb-2 rounded-lg bg-red-50 dark:bg-red-900/20 px-2.5 py-1.5 text-xs text-red-600 dark:text-red-400">
          {file.errorMessage}
        </div>
      )}

      {/* Step details (collapsible) */}
      {hasSteps && !collapsed && (
        <div className="px-3 pb-2.5 space-y-1 border-t border-gray-100 dark:border-gray-800 pt-2">
          {file.done_list?.map((step, i) => (
            <div key={`d-${i}`} className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
              <CheckCircle2 className="w-3.5 h-3.5 text-green-500 shrink-0" />
              <span>{step}</span>
            </div>
          ))}
          {file.running_list?.map((step, i) => (
            <div key={`r-${i}`} className="flex items-center gap-2 text-xs text-violet-600 dark:text-violet-400">
              <Loader2 className="w-3.5 h-3.5 animate-spin shrink-0" />
              <span>{step}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
