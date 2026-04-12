import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import { CopyButton } from '../common';

export function MarkdownRenderer({ content }: { content: string }) {
  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[rehypeHighlight]}
      components={{
        pre({ children, ...props }) {
          const codeText = extractCodeText(children);
          return (
            <div className="relative group my-3">
              <pre
                {...props}
                className="rounded-xl bg-gray-900 dark:bg-gray-950 text-gray-100 p-4 overflow-x-auto text-sm leading-relaxed"
              >
                {children}
              </pre>
              {codeText && (
                <div className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity">
                  <CopyButton content={codeText} size="sm" />
                </div>
              )}
            </div>
          );
        },
        code({ className, children, ...props }) {
          const isInline = !className;
          if (isInline) {
            return (
              <code
                className="px-1.5 py-0.5 rounded bg-gray-100 dark:bg-gray-800 text-pink-600 dark:text-pink-400 text-sm font-mono"
                {...props}
              >
                {children}
              </code>
            );
          }
          return (
            <code className={className} {...props}>
              {children}
            </code>
          );
        },
        a({ href, children, ...props }) {
          return (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-blue-500 hover:text-blue-600 underline"
              {...props}
            >
              {children}
            </a>
          );
        },
        table({ children, ...props }) {
          return (
            <div className="overflow-x-auto my-3">
              <table className="min-w-full border-collapse border border-gray-200 dark:border-gray-700" {...props}>
                {children}
              </table>
            </div>
          );
        },
        th({ children, ...props }) {
          return (
            <th className="border border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800 px-3 py-2 text-left font-medium" {...props}>
              {children}
            </th>
          );
        },
        td({ children, ...props }) {
          return (
            <td className="border border-gray-200 dark:border-gray-700 px-3 py-2" {...props}>
              {children}
            </td>
          );
        },
      }}
    >
      {content}
    </ReactMarkdown>
  );
}

function extractCodeText(children: React.ReactNode): string {
  if (!children) return '';
  if (typeof children === 'string') return children;
  if (Array.isArray(children)) {
    return children.map(extractCodeText).join('');
  }
  if (typeof children === 'object' && children !== null && 'props' in children) {
    return extractCodeText((children as any).props?.children);
  }
  return '';
}
