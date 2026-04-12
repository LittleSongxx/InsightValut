import { useState, useCallback } from 'react';
import { Check, Copy } from 'lucide-react';

export function CopyButton({ content, size = 'sm' }: { content: string; size?: 'sm' | 'md' }) {
  const [copied, setCopied] = useState(false);
  const iconSize = size === 'sm' ? 'w-3.5 h-3.5' : 'w-4 h-4';

  const handleCopy = useCallback(async () => {
    await navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  }, [content]);

  return (
    <button
      onClick={handleCopy}
      className="p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
      title="复制"
    >
      {copied ? <Check className={`${iconSize} text-green-500`} /> : <Copy className={iconSize} />}
    </button>
  );
}
