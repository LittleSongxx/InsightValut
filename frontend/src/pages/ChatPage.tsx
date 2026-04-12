import { useState, useCallback, useRef, useEffect } from 'react';
import type { Message, ProgressData } from '../types';
import { ChatWindow, ChatInput, ProgressPanel, ImportProgressPanel } from '../components';
import type { UploadingFile } from '../components/chat/ChatInput';
import { sendQuery, createSSEConnection, getHistory, uploadFile, getImportStatus } from '../services/api';

function generateId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 9);
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [sessionId, setSessionId] = useState<string>(() => generateId());
  const [isLoading, setIsLoading] = useState(false);
  const [isStreaming, setIsStreaming] = useState(false);
  const [progress, setProgress] = useState<ProgressData | null>(null);
  const [suggestionText, setSuggestionText] = useState('');
  const [uploadingFiles, setUploadingFiles] = useState<UploadingFile[]>([]);
  const [serviceHealth, setServiceHealth] = useState<{ query: boolean; import: boolean }>({ query: false, import: false });
  const eventSourceRef = useRef<EventSource | null>(null);
  const assistantMsgIdRef = useRef<string | null>(null);
  const startTimeRef = useRef<number>(0);
  const uploadPollingRef = useRef<Map<string, ReturnType<typeof setInterval>>>(new Map());

  // Health check on mount
  useEffect(() => {
    const checkHealth = async () => {
      const [queryOk, importOk] = await Promise.all([
        fetch('/api/query/health').then(r => r.ok).catch(() => false),
        fetch('/api/import/health').then(r => r.ok).catch(() => false),
      ]);
      setServiceHealth({ query: queryOk, import: importOk });
    };
    checkHealth();
    const interval = setInterval(checkHealth, 30000); // re-check every 30s
    return () => clearInterval(interval);
  }, []);

  // Load session history on mount if there's a stored session
  useEffect(() => {
    const stored = localStorage.getItem('iv_session_id');
    if (stored) {
      setSessionId(stored);
      getHistory(stored, 50)
        .then((data) => {
          const loaded: Message[] = data.items.map((item, idx) => ({
            id: `hist-${idx}`,
            role: item.role === 'user' ? 'user' : 'assistant',
            content: item.text,
            timestamp: new Date(item.ts * 1000),
          }));
          setMessages(loaded);
        })
        .catch(() => { });
    }
  }, []);

  // Persist session ID
  useEffect(() => {
    localStorage.setItem('iv_session_id', sessionId);
  }, [sessionId]);

  const closeSSE = useCallback(() => {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
  }, []);

  const handleCancel = useCallback(() => {
    closeSSE();
    setIsStreaming(false);
    setIsLoading(false);
    // Mark current assistant message as done
    if (assistantMsgIdRef.current) {
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMsgIdRef.current
            ? { ...m, isStreaming: false, endTime: Date.now() }
            : m,
        ),
      );
    }
  }, [closeSSE]);

  const handleSend = useCallback(
    async (text: string) => {
      // Add user message
      const userMsg: Message = {
        id: generateId(),
        role: 'user',
        content: text,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, userMsg]);
      setIsLoading(true);
      setProgress(null);

      try {
        // Send query (stream mode)
        const res = await sendQuery(text, sessionId, true);
        const sid = res.session_id || sessionId;
        if (sid !== sessionId) setSessionId(sid);

        // Create assistant message placeholder
        const assistantId = generateId();
        assistantMsgIdRef.current = assistantId;
        startTimeRef.current = Date.now();
        const assistantMsg: Message = {
          id: assistantId,
          role: 'assistant',
          content: '',
          timestamp: new Date(),
          isStreaming: true,
          startTime: startTimeRef.current,
        };
        setMessages((prev) => [...prev, assistantMsg]);
        setIsStreaming(true);

        // Connect SSE
        const es = createSSEConnection(sid);
        eventSourceRef.current = es;

        es.addEventListener('progress', (e: MessageEvent) => {
          try {
            const data = JSON.parse(e.data);
            setProgress(data);
          } catch { }
        });

        es.addEventListener('delta', (e: MessageEvent) => {
          try {
            const data = JSON.parse(e.data);
            const chunk = data.chunk || data.content || '';
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, content: m.content + chunk }
                  : m,
              ),
            );
          } catch { }
        });

        es.addEventListener('final', (e: MessageEvent) => {
          try {
            const data = JSON.parse(e.data);
            const answer = data.answer || data.content || '';
            const images = data.image_urls || [];
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? {
                    ...m,
                    content: answer || m.content,
                    isStreaming: false,
                    endTime: Date.now(),
                    images: images.length > 0 ? images : m.images,
                  }
                  : m,
              ),
            );
          } catch { }
          setIsStreaming(false);
          setIsLoading(false);
          setProgress((prev) => prev ? { ...prev, status: 'completed' } : null);
          // Auto-hide progress panel after a short delay
          setTimeout(() => setProgress(null), 3000);
          closeSSE();
        });

        es.addEventListener('error', (e: MessageEvent) => {
          try {
            const data = JSON.parse((e as any).data || '{}');
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, content: m.content || `错误: ${data.error || '未知错误'}`, isStreaming: false, endTime: Date.now() }
                  : m,
              ),
            );
          } catch { }
          setIsStreaming(false);
          setIsLoading(false);
          setProgress(null);
          closeSSE();
        });

        es.onerror = () => {
          setIsStreaming(false);
          setIsLoading(false);
          setProgress(null);
          closeSSE();
        };
      } catch (err: any) {
        const isNetworkError = err?.message === 'Failed to fetch' || err?.message?.includes('fetch');
        const errorMsg: Message = {
          id: generateId(),
          role: 'assistant',
          content: isNetworkError
            ? `请求失败: 无法连接到查询服务。请确认后端服务（app-query）是否已启动，并检查是否在 Vite dev server（localhost:3000）环境下运行。`
            : `请求失败: ${err.message || '网络错误'}`,
          timestamp: new Date(),
        };
        setMessages((prev) => [...prev, errorMsg]);
        setIsLoading(false);
      }
    },
    [sessionId, closeSSE],
  );

  const handleNewChat = useCallback(() => {
    closeSSE();
    setMessages([]);
    setProgress(null);
    setIsLoading(false);
    setIsStreaming(false);
    const newSid = generateId();
    setSessionId(newSid);
  }, [closeSSE]);

  // --- File upload from chat input ---
  useEffect(() => {
    return () => {
      uploadPollingRef.current.forEach((t) => clearInterval(t));
    };
  }, []);

  const startUploadPolling = useCallback((fileId: string, taskId: string, fileName: string) => {
    const timer = setInterval(async () => {
      try {
        const res = await getImportStatus(taskId);
        const mappedStatus: UploadingFile['status'] =
          res.status === 'completed' ? 'completed' :
            res.status === 'failed' ? 'failed' : 'processing';
        setUploadingFiles((prev) =>
          prev.map((f) =>
            f.id === fileId
              ? {
                ...f,
                status: mappedStatus,
                errorMessage: res.error_message || f.errorMessage,
                done_list: res.done_list || [],
                running_list: res.running_list || [],
              }
              : f,
          ),
        );
        if (res.status === 'completed' || res.status === 'failed') {
          clearInterval(uploadPollingRef.current.get(fileId));
          uploadPollingRef.current.delete(fileId);
          if (res.status === 'completed') {
            const sysMsg: Message = {
              id: generateId(),
              role: 'assistant',
              content: `${fileName} 已成功导入产品知识库，您现在可以针对该产品进行提问。`,
              timestamp: new Date(),
            };
            setMessages((prev) => [...prev, sysMsg]);
          }
        }
      } catch {
        // ignore polling errors
      }
    }, 2000);
    uploadPollingRef.current.set(fileId, timer);
  }, []);

  const handleFileUpload = useCallback(
    async (file: File) => {
      const fileId = `uf-${Date.now()}-${Math.random().toString(36).slice(2, 7)}`;
      const newUf: UploadingFile = {
        id: fileId,
        name: file.name,
        status: 'uploading',
        progress: 0,
      };
      setUploadingFiles((prev) => [...prev, newUf]);

      try {
        const res = await uploadFile(file, (pct) => {
          setUploadingFiles((prev) =>
            prev.map((f) => (f.id === fileId ? { ...f, progress: pct } : f)),
          );
        });
        setUploadingFiles((prev) =>
          prev.map((f) =>
            f.id === fileId
              ? { ...f, status: 'processing', progress: 100, taskId: res.task_id }
              : f,
          ),
        );
        startUploadPolling(fileId, res.task_id, file.name);
      } catch (err: any) {
        const isNetworkError = err?.message === 'Failed to fetch' || err?.message?.includes('fetch') || err?.message?.includes('network');
        setUploadingFiles((prev) =>
          prev.map((f) =>
            f.id === fileId
              ? {
                  ...f,
                  status: 'failed',
                  errorMessage: isNetworkError
                    ? '无法连接到导入服务，请确认后端服务（app-import）是否已启动'
                    : (err?.message || '上传失败'),
                }
              : f,
          ),
        );
      }
    },
    [startUploadPolling],
  );

  const handleRemoveFile = useCallback((id: string) => {
    setUploadingFiles((prev) => prev.filter((f) => f.id !== id));
    const timer = uploadPollingRef.current.get(id);
    if (timer) {
      clearInterval(timer);
      uploadPollingRef.current.delete(id);
    }
  }, []);

  return (
    <>
      <ChatWindow
        messages={messages}
        isLoading={isLoading}
        onSuggestionClick={(t) => setSuggestionText(t)}
      />
      {progress && (progress.status === 'processing' || progress.done_list.length > 0) && (
        <ProgressPanel progress={progress} />
      )}
      <div className="p-4 max-w-4xl w-full mx-auto space-y-3">
        {(!serviceHealth.query || !serviceHealth.import) && (
          <div className="bg-amber-50 border border-amber-200 rounded-lg px-4 py-3 text-sm text-amber-800">
            <div className="font-medium mb-1">⚠ 后端服务未连接</div>
            <ul className="list-disc list-inside space-y-0.5">
              {!serviceHealth.query && <li>查询服务（app-query）未运行 — 无法提问</li>}
              {!serviceHealth.import && <li>导入服务（app-import）未运行 — 无法上传文档</li>}
            </ul>
          </div>
        )}
        {uploadingFiles.some((f) => f.status === 'uploading' || f.status === 'processing' || ((f.done_list?.length ?? 0) > 0)) && (
          <ImportProgressPanel files={uploadingFiles} />
        )}
        <ChatInput
          onSend={handleSend}
          onCancel={handleCancel}
          onFileUpload={handleFileUpload}
          disabled={isLoading}
          isStreaming={isStreaming}
          externalValue={suggestionText}
          onExternalValueConsumed={() => setSuggestionText('')}
          uploadingFiles={uploadingFiles}
          onRemoveFile={handleRemoveFile}
        />
        <div className="text-center text-xs text-gray-400 mt-2">
          InsightVault 可能会犯错，请注意核实重要信息。
        </div>
      </div>
    </>
  );
}
