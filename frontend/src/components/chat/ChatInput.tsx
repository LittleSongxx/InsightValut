import { useState, useRef, useEffect, useCallback, type KeyboardEvent, type ChangeEvent } from 'react';
import { SendHorizontal, Square, Plus, FileText, X, Loader2, CheckCircle2 } from 'lucide-react';

export interface UploadingFile {
  id: string;
  name: string;
  status: 'uploading' | 'processing' | 'completed' | 'failed';
  progress: number;
  taskId?: string;
  errorMessage?: string;
  done_list?: string[];
  running_list?: string[];
}

export interface ChatInputProps {
  onSend: (message: string) => void;
  onCancel?: () => void;
  onFileUpload?: (file: File) => void;
  disabled?: boolean;
  isStreaming?: boolean;
  placeholder?: string;
  externalValue?: string;
  onExternalValueConsumed?: () => void;
  uploadingFiles?: UploadingFile[];
  onRemoveFile?: (id: string) => void;
}

export function ChatInput({
  onSend,
  onCancel,
  onFileUpload,
  disabled = false,
  isStreaming = false,
  placeholder = '输入您的问题...',
  externalValue,
  onExternalValueConsumed,
  uploadingFiles = [],
  onRemoveFile,
}: ChatInputProps) {
  const [input, setInput] = useState('');
  const [isComposing, setIsComposing] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const dragCounterRef = useRef(0);

  useEffect(() => {
    if (externalValue !== undefined && externalValue !== '') {
      const frameId = window.requestAnimationFrame(() => {
        setInput(externalValue);
        onExternalValueConsumed?.();
        textareaRef.current?.focus();
      });
      return () => window.cancelAnimationFrame(frameId);
    }
  }, [externalValue, onExternalValueConsumed]);

  const handleSend = () => {
    if (input.trim() && !disabled && !isStreaming) {
      onSend(input.trim());
      setInput('');
      if (textareaRef.current) textareaRef.current.style.height = 'auto';
    }
  };

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey && !isComposing) {
      e.preventDefault();
      handleSend();
    }
  };

  const handleInput = (e: ChangeEvent<HTMLTextAreaElement>) => {
    setInput(e.target.value);
    const textarea = e.target;
    textarea.style.height = 'auto';
    textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
  };

  const handleFiles = useCallback(
    (files: FileList | File[]) => {
      if (!onFileUpload) return;
      Array.from(files).forEach((f) => {
        const ext = f.name.split('.').pop()?.toLowerCase();
        if (['pdf', 'md', 'txt'].includes(ext || '')) {
          onFileUpload(f);
        }
      });
    },
    [onFileUpload],
  );

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounterRef.current += 1;
    if (dragCounterRef.current === 1) setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    dragCounterRef.current -= 1;
    if (dragCounterRef.current === 0) setIsDragging(false);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      e.stopPropagation();
      dragCounterRef.current = 0;
      setIsDragging(false);
      if (e.dataTransfer.files.length > 0) {
        handleFiles(e.dataTransfer.files);
      }
    },
    [handleFiles],
  );

  const hasActiveUploads = uploadingFiles.some((f) => f.status === 'uploading' || f.status === 'processing');
  const canSend = input.trim() && !disabled && !isStreaming && !hasActiveUploads;

  return (
    <div
      className={`relative rounded-2xl border transition-all ${isDragging
        ? 'border-violet-400 bg-violet-50/50 dark:bg-violet-900/20 ring-2 ring-violet-500/20'
        : 'border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 focus-within:ring-2 focus-within:ring-violet-500/20'
        } shadow-sm`}
      onDragEnter={handleDragEnter}
      onDragLeave={handleDragLeave}
      onDragOver={handleDragOver}
      onDrop={handleDrop}
    >
      {/* File chips */}
      {uploadingFiles.length > 0 && (
        <div className="flex flex-wrap gap-2 px-3 pt-2.5 pb-1">
          {uploadingFiles.map((uf) => (
            <FileChip key={uf.id} file={uf} onRemove={onRemoveFile} />
          ))}
        </div>
      )}

      {/* Drag overlay hint */}
      {isDragging && (
        <div className="absolute inset-0 z-10 flex items-center justify-center rounded-2xl bg-violet-50/90 dark:bg-violet-900/80 pointer-events-none">
          <p className="text-violet-600 dark:text-violet-300 font-medium text-sm">
            松开以上传文件到知识库
          </p>
        </div>
      )}

      {/* Input row */}
      <div className="flex items-end gap-1.5 p-2">
        {/* + button */}
        <button
          type="button"
          onClick={() => fileInputRef.current?.click()}
          className="p-2 rounded-lg text-gray-400 hover:text-violet-500 hover:bg-violet-50 dark:hover:bg-violet-900/30 transition-colors shrink-0"
          title="上传文件到知识库"
        >
          <Plus className="w-5 h-5" />
        </button>
        <input
          ref={fileInputRef}
          type="file"
          multiple
          accept=".pdf,.md,.txt"
          className="hidden"
          onChange={(e) => {
            if (e.target.files) handleFiles(e.target.files);
            e.target.value = '';
          }}
        />

        <textarea
          ref={textareaRef}
          value={input}
          onChange={handleInput}
          onKeyDown={handleKeyDown}
          onCompositionStart={() => setIsComposing(true)}
          onCompositionEnd={() => setIsComposing(false)}
          placeholder={placeholder}
          rows={1}
          className="flex-1 max-h-[200px] py-1.5 px-2 bg-transparent border-none focus:ring-0 focus:outline-none resize-none text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 text-sm leading-relaxed scrollbar-hide"
        />
        {isStreaming ? (
          <button
            onClick={onCancel}
            className="p-2 rounded-lg transition-all duration-200 bg-red-500 text-white hover:bg-red-600 shadow-sm shrink-0"
            title="停止生成"
          >
            <Square className="w-5 h-5" />
          </button>
        ) : (
          <button
            onClick={handleSend}
            disabled={!canSend}
            className={`p-2 rounded-lg transition-all duration-200 shrink-0 ${canSend
              ? 'bg-violet-500 text-white hover:bg-violet-600 shadow-sm'
              : 'bg-gray-100 dark:bg-gray-700 text-gray-400 dark:text-gray-500 cursor-not-allowed'
              }`}
            title="发送"
          >
            <SendHorizontal className="w-5 h-5" />
          </button>
        )}
      </div>
    </div>
  );
}

function FileChip({ file, onRemove }: { file: UploadingFile; onRemove?: (id: string) => void }) {
  const statusIcon = {
    uploading: <Loader2 className="w-3.5 h-3.5 text-blue-500 animate-spin shrink-0" />,
    processing: <Loader2 className="w-3.5 h-3.5 text-violet-500 animate-spin shrink-0" />,
    completed: <CheckCircle2 className="w-3.5 h-3.5 text-green-500 shrink-0" />,
    failed: <X className="w-3.5 h-3.5 text-red-500 shrink-0" />,
  }[file.status];

  return (
    <div
      className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs transition-colors ${file.status === 'failed'
        ? 'bg-red-50 dark:bg-red-900/20 text-red-600 dark:text-red-400'
        : file.status === 'completed'
          ? 'bg-green-50 dark:bg-green-900/20 text-green-700 dark:text-green-400'
          : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300'
        }`}
      title={file.errorMessage || file.name}
    >
      <FileText className="w-3.5 h-3.5 shrink-0" />
      <span className="max-w-[120px] truncate">{file.name}</span>
      {statusIcon}
      {file.status === 'uploading' && (
        <span className="text-[10px] text-gray-400">{file.progress}%</span>
      )}
      {(file.status === 'completed' || file.status === 'failed') && onRemove && (
        <button
          type="button"
          onClick={() => onRemove(file.id)}
          className="p-0.5 rounded hover:bg-black/10 dark:hover:bg-white/10"
        >
          <X className="w-3 h-3" />
        </button>
      )}
    </div>
  );
}
