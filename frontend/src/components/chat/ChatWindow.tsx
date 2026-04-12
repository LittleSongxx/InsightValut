import { useEffect, useRef, useState, useCallback } from 'react';
import { BookOpen, Search, Sparkles } from 'lucide-react';
import type { Message } from '../../types';
import { MessageBubble } from './MessageBubble';

export interface ChatWindowProps {
  messages: Message[];
  isLoading: boolean;
  onSuggestionClick?: (text: string) => void;
}

export function ChatWindow({ messages, isLoading, onSuggestionClick }: ChatWindowProps) {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const [isNearBottom, setIsNearBottom] = useState(true);
  const [isUserScrolling, setIsUserScrolling] = useState(false);

  const checkIsNearBottom = useCallback(() => {
    const container = scrollContainerRef.current;
    if (!container) return true;
    return container.scrollHeight - container.scrollTop - container.clientHeight < 100;
  }, []);

  const handleScroll = useCallback(() => {
    if (!isUserScrolling) {
      setIsUserScrolling(true);
      setTimeout(() => setIsUserScrolling(false), 1000);
    }
    setIsNearBottom(checkIsNearBottom());
  }, [isUserScrolling, checkIsNearBottom]);

  useEffect(() => {
    if (isNearBottom && !isUserScrolling && messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages, isNearBottom, isUserScrolling]);

  useEffect(() => {
    const container = scrollContainerRef.current;
    if (container) {
      container.addEventListener('scroll', handleScroll, { passive: true });
      return () => container.removeEventListener('scroll', handleScroll);
    }
  }, [handleScroll]);

  const isEmpty = messages.length === 0;

  return (
    <div
      ref={scrollContainerRef}
      className={`flex-1 p-4 md:p-6 bg-gradient-to-b from-white to-gray-50 dark:from-gray-900 dark:to-gray-950 ${isEmpty ? 'overflow-y-hidden' : 'overflow-y-auto'}`}
    >
      {isEmpty ? (
        <EmptyState onSuggestionClick={onSuggestionClick} />
      ) : (
        <div className="max-w-3xl mx-auto w-full">
          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}
          {isLoading && messages.length > 0 && messages[messages.length - 1].role === 'user' && <LoadingIndicator />}
        </div>
      )}
      {!isEmpty && <div ref={messagesEndRef} className="h-4" />}
    </div>
  );
}

function EmptyState({ onSuggestionClick }: { onSuggestionClick?: (text: string) => void }) {
  return (
    <div className="flex flex-col items-center justify-center h-full w-full text-gray-500 dark:text-gray-400 animate-in fade-in duration-500 px-4">
      <section className="empty-state-hero relative flex-1 flex flex-col items-center justify-center overflow-hidden">
        <div className="relative group w-full px-4">
          <div className="flex flex-col items-center gap-2">
            <div className="w-20 h-20 rounded-2xl bg-gradient-to-br from-violet-400 to-violet-600 flex items-center justify-center shadow-lg">
              <Sparkles className="w-10 h-10 text-white" />
            </div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-violet-600 to-blue-500 bg-clip-text text-transparent mt-4">
              InsightVault
            </h1>
            <p className="text-sm text-gray-500 dark:text-gray-400">智能产品知识库助手</p>
          </div>
        </div>
      </section>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 w-full max-w-3xl mb-8">
        <FeatureCard icon={<Search className="w-5 h-5" />} title="产品检索" description="从产品手册中精准检索答案" />
        <FeatureCard icon={<BookOpen className="w-5 h-5" />} title="多源融合" description="向量+图谱+网络多路搜索" />
        <FeatureCard icon={<Sparkles className="w-5 h-5" />} title="智能分析" description="CRAG 质量检查与幻觉自检" />
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 w-full max-w-2xl">
        <SuggestionChip text="这个产品的技术参数是什么？" onClick={onSuggestionClick} />
        <SuggestionChip text="如何安装和调试这台设备？" onClick={onSuggestionClick} />
        <SuggestionChip text="常见故障及排除方法" onClick={onSuggestionClick} />
        <SuggestionChip text="这两个型号有什么区别？" onClick={onSuggestionClick} />
      </div>
    </div>
  );
}

function FeatureCard({ icon, title, description }: { icon: React.ReactNode; title: string; description: string }) {
  return (
    <div className="p-5 bg-white dark:bg-gray-800 border border-gray-200/60 dark:border-gray-700/60 rounded-xl text-center shadow-sm hover:shadow-lg hover:shadow-violet-500/10 hover:-translate-y-1 transition-all duration-300">
      <div className="w-11 h-11 bg-gradient-to-br from-violet-100 to-violet-50 dark:from-violet-900/40 dark:to-violet-900/20 rounded-lg flex items-center justify-center mx-auto mb-3 text-violet-500">
        {icon}
      </div>
      <h3 className="font-semibold text-gray-800 dark:text-gray-100 mb-1.5">{title}</h3>
      <p className="text-xs text-gray-500 dark:text-gray-400">{description}</p>
    </div>
  );
}

function SuggestionChip({ text, onClick }: { text: string; onClick?: (text: string) => void }) {
  return (
    <button
      className="flex items-center gap-3 px-4 py-3.5 bg-white dark:bg-gray-800 border border-gray-200/60 dark:border-gray-700/60 rounded-xl text-sm text-left hover:border-violet-400 dark:hover:border-violet-600 hover:shadow-lg hover:shadow-violet-500/10 hover:-translate-y-0.5 transition-all duration-300 text-gray-700 dark:text-gray-300 group"
      onClick={() => onClick?.(text)}
    >
      <span className="text-lg group-hover:scale-125 group-hover:rotate-3 transition-all duration-300">💡</span>
      <span className="group-hover:text-violet-600 dark:group-hover:text-violet-400 transition-colors">{text}</span>
    </button>
  );
}

function LoadingIndicator() {
  return (
    <div className="flex gap-4 mb-6">
      <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-violet-400 to-violet-500 flex items-center justify-center shrink-0 shadow-sm">
        <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
      </div>
      <div className="space-y-2 pt-2">
        <span className="text-sm text-gray-500 dark:text-gray-400 animate-pulse">InsightVault 正在思考...</span>
      </div>
    </div>
  );
}
