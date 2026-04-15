import { useState, useEffect } from 'react';
import { Clock, Loader2, Sparkles, ImageIcon } from 'lucide-react';
import type { Message } from '../../types';
import { MarkdownRenderer } from './MarkdownRenderer';
import { CopyButton } from '../common';
import { normalizeAssetUrls } from '../../utils/media';

function formatCoverageScore(value?: number) {
  if (typeof value !== 'number' || Number.isNaN(value)) return null;
  return `${(value * 100).toFixed(0)}%`;
}

function AgenticTracePanel({ message }: { message: Message }) {
  const metadata = message.metadata;
  if (!metadata) return null;

  const coverage = metadata.evidence_coverage_summary;
  const rescue = metadata.rescue_plan;
  const contextExpansion = metadata.context_expansion_summary;
  const answerPlan = metadata.answer_plan;
  const queryType = metadata.query_type;
  const coverageScore = formatCoverageScore(coverage?.coverage_score);
  const rescueSteps = rescue?.steps?.filter(Boolean) ?? [];
  const focusTerms = metadata.query_focus_terms?.filter(Boolean) ?? [];
  const missingTerms = coverage?.missing_focus_terms?.filter(Boolean) ?? [];
  const hasMeaningfulContent = Boolean(
    queryType ||
    typeof metadata.graph_preferred === 'boolean' ||
    coverageScore ||
    rescue?.action ||
    (contextExpansion?.expanded_docs ?? 0) > 0 ||
    answerPlan?.response_format ||
    focusTerms.length > 0,
  );

  if (!hasMeaningfulContent) return null;

  return (
    <div className="mt-3 w-full max-w-4xl rounded-2xl border border-sky-200/80 bg-sky-50/80 px-4 py-3 shadow-sm dark:border-sky-900/70 dark:bg-slate-900/70">
      <div className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-[0.16em] text-sky-700 dark:text-sky-300">
        <Sparkles className="h-3.5 w-3.5" />
        Agentic Trace
      </div>
      <div className="flex flex-wrap gap-2 text-xs text-slate-700 dark:text-slate-300">
        {queryType && (
          <span className="rounded-full bg-white/90 px-2.5 py-1 shadow-sm dark:bg-slate-800">
            题型：{queryType}
          </span>
        )}
        {typeof metadata.graph_preferred === 'boolean' && (
          <span className="rounded-full bg-white/90 px-2.5 py-1 shadow-sm dark:bg-slate-800">
            图优先：{metadata.graph_preferred ? '是' : '否'}
          </span>
        )}
        {coverageScore && (
          <span className="rounded-full bg-white/90 px-2.5 py-1 shadow-sm dark:bg-slate-800">
            证据覆盖：{coverageScore}
          </span>
        )}
        {answerPlan?.response_format && (
          <span className="rounded-full bg-white/90 px-2.5 py-1 shadow-sm dark:bg-slate-800">
            回答格式：{answerPlan.response_format}
          </span>
        )}
        {(contextExpansion?.expanded_docs ?? 0) > 0 && (
          <span className="rounded-full bg-white/90 px-2.5 py-1 shadow-sm dark:bg-slate-800">
            上下文扩展：{contextExpansion?.expanded_docs}/{contextExpansion?.candidate_docs ?? contextExpansion?.expanded_docs}
          </span>
        )}
        {rescue?.action && rescue.action !== 'none' && (
          <span className="rounded-full bg-white/90 px-2.5 py-1 shadow-sm dark:bg-slate-800">
            补救动作：{rescue.action}
          </span>
        )}
      </div>

      {focusTerms.length > 0 && (
        <div className="mt-3 text-xs text-slate-600 dark:text-slate-400">
          焦点词：{focusTerms.slice(0, 6).join('、')}
        </div>
      )}

      {missingTerms.length > 0 && (
        <div className="mt-2 text-xs text-amber-700 dark:text-amber-300">
          尚未完全覆盖：{missingTerms.slice(0, 5).join('、')}
        </div>
      )}

      {rescueSteps.length > 0 && (
        <div className="mt-2 text-xs text-slate-600 dark:text-slate-400">
          补救步骤：{rescueSteps.join(' → ')}
        </div>
      )}

      {metadata.clarification_reason && (
        <div className="mt-2 text-xs text-slate-600 dark:text-slate-400">
          澄清原因：{metadata.clarification_reason}
        </div>
      )}
    </div>
  );
}

export function MessageBubble({ message }: { message: Message }) {
  const isUser = message.role === 'user';
  const hasText = !!(message.content && message.content.trim().length > 0);
  const normalizedImages = normalizeAssetUrls(message.images);

  const totalDuration = message.startTime && message.endTime
    ? message.endTime - message.startTime
    : undefined;

  const [elapsedTime, setElapsedTime] = useState(0);

  useEffect(() => {
    if (!message.isStreaming || !message.startTime) {
      return;
    }
    const interval = setInterval(() => {
      setElapsedTime(Date.now() - message.startTime!);
    }, 100);
    return () => clearInterval(interval);
  }, [message.isStreaming, message.startTime]);

  const formatDuration = (ms: number) => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };

  return (
    <div className={`flex mb-6 ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`flex flex-col w-full ${isUser ? 'items-end' : 'items-start'}`}>
        {/* User images */}
        {isUser && normalizedImages.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-2">
            {normalizedImages.map((img, idx) => (
              <img key={idx} src={img} alt={`上传图片 ${idx + 1}`} className="max-w-[200px] max-h-[200px] rounded-lg object-cover" />
            ))}
          </div>
        )}

        {/* Message content */}
        {hasText && (
          <div
            className={`text-sm leading-relaxed break-words ${isUser
              ? 'bg-gradient-to-br from-blue-500 to-blue-600 text-white px-4 py-3 rounded-3xl shadow-sm max-w-[80%]'
              : 'w-full max-w-4xl rounded-3xl border border-gray-200/70 dark:border-gray-700/70 bg-white/95 dark:bg-gray-900/80 px-5 py-4 shadow-sm'
              }`}
          >
            {isUser ? message.content.trim() : (
              <div className="space-y-4">
                <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-[0.18em] text-violet-500 dark:text-violet-300">
                  <Sparkles className="h-3.5 w-3.5" />
                  InsightVault
                </div>
                <MarkdownRenderer content={message.content.trim()} />
              </div>
            )}
          </div>
        )}

        {!isUser && normalizedImages.length > 0 && (
          <div className="mt-3 w-full max-w-4xl rounded-2xl border border-gray-200/70 bg-white/80 p-4 shadow-sm dark:border-gray-700/70 dark:bg-gray-900/60">
            <div className="mb-3 flex items-center gap-2 text-sm font-medium text-gray-700 dark:text-gray-200">
              <ImageIcon className="h-4 w-4 text-violet-500" />
              相关图片
            </div>
            <div className="flex flex-wrap gap-3">
              {normalizedImages.map((img, idx) => (
                <a key={idx} href={img} target="_blank" rel="noreferrer" className="block overflow-hidden rounded-2xl border border-gray-200 bg-white shadow-sm transition-transform hover:-translate-y-0.5 dark:border-gray-700 dark:bg-gray-950">
                  <img
                    src={img}
                    alt={`回答图片 ${idx + 1}`}
                    className="h-40 w-40 object-cover sm:h-48 sm:w-48"
                  />
                </a>
              ))}
            </div>
          </div>
        )}

        {!isUser && <AgenticTracePanel message={message} />}

        {/* Streaming cursor */}
        {!isUser && message.isStreaming && !hasText && (
          <div className="flex gap-3 items-center">
            <div className="w-8 h-8 rounded-xl bg-gradient-to-br from-violet-400 to-violet-500 flex items-center justify-center shrink-0 shadow-sm">
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            </div>
            <span className="text-sm text-gray-500 dark:text-gray-400 animate-pulse">InsightVault 正在思考...</span>
          </div>
        )}

        {/* Sources */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="mt-3 w-full max-w-4xl rounded-2xl border border-gray-200/70 bg-white/80 px-4 py-3 shadow-sm dark:border-gray-700/70 dark:bg-gray-900/60">
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-1 font-medium">参考来源：</p>
            <ul className="space-y-0.5">
              {message.sources.map((src, idx) => (
                <li key={idx} className="text-xs text-gray-600 dark:text-gray-400 flex items-center gap-1.5">
                  <span className="w-1.5 h-1.5 rounded-full bg-green-400 shrink-0" />
                  {src}
                </li>
              ))}
            </ul>
          </div>
        )}

        {/* Timestamp + Duration */}
        <div className={`text-xs mt-2 flex items-center gap-2 flex-wrap ${isUser ? 'text-gray-400' : 'text-gray-500 dark:text-gray-400'}`}>
          {hasText && isUser && <CopyButton content={message.content.trim()} size="sm" />}
          <span>{message.timestamp.toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })}</span>

          {!isUser && message.isStreaming && elapsedTime > 0 && (
            <span className="inline-flex items-center gap-1 text-blue-500 dark:text-blue-400">
              <Loader2 className="w-3 h-3 animate-spin" />
              {formatDuration(elapsedTime)}
            </span>
          )}

          {!isUser && !message.isStreaming && totalDuration !== undefined && (
            <span className="inline-flex items-center gap-1">
              <Clock className="w-3 h-3" />
              耗时 {formatDuration(totalDuration)}
            </span>
          )}

          {hasText && !isUser && <CopyButton content={message.content.trim()} size="sm" />}
        </div>
      </div>
    </div>
  );
}
