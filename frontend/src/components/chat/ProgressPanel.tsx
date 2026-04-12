import { CheckCircle2, Loader2 } from 'lucide-react';
import type { ProgressData } from '../../types';

export function ProgressPanel({ progress }: { progress: ProgressData | null }) {
  if (!progress) return null;
  const { status, done_list, running_list } = progress;
  if (status === 'completed' && done_list.length === 0) return null;

  return (
    <div className="mx-auto max-w-3xl w-full mb-4">
      <div className="rounded-xl border border-gray-200 dark:border-gray-800 bg-white dark:bg-gray-900/60 p-3 shadow-sm">
        <div className="flex items-center gap-2 mb-2">
          {status === 'processing' ? (
            <Loader2 className="w-4 h-4 text-violet-500 animate-spin" />
          ) : status === 'completed' ? (
            <CheckCircle2 className="w-4 h-4 text-green-500" />
          ) : null}
          <span className="text-xs font-semibold text-gray-600 dark:text-gray-300">
            {status === 'processing' ? '处理中...' : status === 'completed' ? '处理完成' : status === 'failed' ? '处理失败' : '等待中'}
          </span>
        </div>

        {/* Done items */}
        {done_list.length > 0 && (
          <div className="space-y-1">
            {done_list.map((item, i) => (
              <div key={i} className="flex items-center gap-2 text-xs text-gray-500 dark:text-gray-400">
                <CheckCircle2 className="w-3.5 h-3.5 text-green-500 shrink-0" />
                <span>{item}</span>
              </div>
            ))}
          </div>
        )}

        {/* Running items */}
        {running_list.length > 0 && (
          <div className="space-y-1 mt-1">
            {running_list.map((item, i) => (
              <div key={i} className="flex items-center gap-2 text-xs text-violet-600 dark:text-violet-400">
                <Loader2 className="w-3.5 h-3.5 animate-spin shrink-0" />
                <span>{item}</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
