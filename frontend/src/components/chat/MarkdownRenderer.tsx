import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeHighlight from 'rehype-highlight';
import type { Components } from 'react-markdown';
import { CopyButton } from '../common';
import { normalizeAssetUrl } from '../../utils/media';

import 'highlight.js/styles/github-dark.css';

const components: Components = {
  h1: ({ children }) => (
    <h1 className="text-2xl font-bold text-gray-900 dark:text-white mt-2 mb-3">{children}</h1>
  ),
  h2: ({ children }) => (
    <h2 className="text-xl font-semibold text-gray-900 dark:text-white mt-5 mb-3">{children}</h2>
  ),
  h3: ({ children }) => (
    <h3 className="text-lg font-semibold text-gray-900 dark:text-white mt-4 mb-2">{children}</h3>
  ),
  h4: ({ children }) => (
    <h4 className="text-base font-semibold text-gray-900 dark:text-white mt-3 mb-2">{children}</h4>
  ),
  p: ({ children }) => (
    <p className="text-[15px] leading-7 text-gray-700 dark:text-gray-200 my-2">{children}</p>
  ),
  ul: ({ children }) => (
    <ul className="my-3 ml-5 list-disc space-y-1.5 text-gray-700 dark:text-gray-200">{children}</ul>
  ),
  ol: ({ children }) => (
    <ol className="my-3 ml-5 list-decimal space-y-1.5 text-gray-700 dark:text-gray-200">{children}</ol>
  ),
  li: ({ children }) => (
    <li className="leading-7">{children}</li>
  ),
  blockquote: ({ children }) => (
    <blockquote className="my-4 rounded-r-2xl border-l-4 border-violet-400 bg-violet-50/80 px-4 py-3 text-gray-700 dark:border-violet-500 dark:bg-violet-900/20 dark:text-gray-200">{children}</blockquote>
  ),
  pre({ children, ...props }) {
    const codeText = extractCodeText(children);
    return (
      <div className="relative group my-4">
        <pre
          {...props}
          className="overflow-x-auto rounded-2xl bg-gray-900 p-4 text-sm leading-relaxed text-gray-100 shadow-sm"
        >
          {children}
        </pre>
        {codeText && (
          <div className="absolute top-2 right-2 opacity-0 transition-opacity group-hover:opacity-100">
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
          className="rounded-md bg-gray-100 px-1.5 py-0.5 font-mono text-sm text-pink-600 dark:bg-gray-800 dark:text-pink-400"
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
        className="font-medium text-violet-600 underline decoration-violet-300 underline-offset-4 hover:text-violet-700 dark:text-violet-400 dark:decoration-violet-600 dark:hover:text-violet-300"
        {...props}
      >
        {children}
      </a>
    );
  },
  img({ src, alt, ...props }) {
    const normalizedSrc = normalizeAssetUrl(src);
    return (
      <img
        src={normalizedSrc}
        alt={alt || '插图'}
        className="my-4 max-h-[28rem] rounded-2xl border border-gray-200 object-contain shadow-sm dark:border-gray-700"
        {...props}
      />
    );
  },
  table({ children, ...props }) {
    return (
      <div className="my-4 overflow-x-auto rounded-2xl border border-gray-200 dark:border-gray-700">
        <table className="min-w-full border-collapse bg-white dark:bg-gray-950" {...props}>
          {children}
        </table>
      </div>
    );
  },
  thead: ({ children }) => (
    <thead className="bg-gray-50 dark:bg-gray-800">{children}</thead>
  ),
  th({ children, ...props }) {
    return (
      <th className="border-b border-gray-200 px-3 py-2 text-left font-semibold text-gray-700 dark:border-gray-700 dark:text-gray-200" {...props}>
        {children}
      </th>
    );
  },
  td({ children, ...props }) {
    return (
      <td className="border-b border-gray-100 px-3 py-2 text-sm text-gray-700 dark:border-gray-800 dark:text-gray-200" {...props}>
        {children}
      </td>
    );
  },
  hr: () => (
    <hr className="my-5 border-gray-200 dark:border-gray-700" />
  ),
  strong: ({ children }) => (
    <strong className="font-semibold text-gray-900 dark:text-white">{children}</strong>
  ),
  em: ({ children }) => (
    <em className="italic">{children}</em>
  ),
};

const normalizeMarkdown = (content: string) => {
  const normalized = content
    .replace(/\r\n?/g, '\n')
    .trim();

  const replacements: Array<[RegExp, string]> = [
    [/(\*\*[^*\n]+?\*\*)\s*([*-]\s+|\d+\.\s+)/g, '$1\n$2'],
    [/([。！？!?:：])\s*([*-]\s+|\d+\.\s+)/g, '$1\n$2'],
    [/([^\n])(\#{1,}\s*)/g, '$1\n$2'],
    [/([^\n])((?:[*-]|\d+\.)\s+\*\*)/g, '$1\n$2'],
    [/(\*\*[^*]+\*\*)(?=\*\*)/g, '$1\n'],
  ];

  return replacements.reduce(
    (text, [pattern, replacement]) => text.replace(pattern, replacement),
    normalized,
  );
};

export function MarkdownRenderer({ content }: { content: string }) {
  if (!content) return null;
  const normalized = normalizeMarkdown(content);

  return (
    <ReactMarkdown
      remarkPlugins={[remarkGfm]}
      rehypePlugins={[rehypeHighlight]}
      components={components}
    >
      {normalized}
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
    return extractCodeText((children as { props?: { children?: React.ReactNode } }).props?.children);
  }
  return '';
}
