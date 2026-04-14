import { useCallback, useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
  MessageSquarePlus,
  BookOpen,
  Upload,
  BarChart3,
  Moon,
  Sun,
  ChevronLeft,
  ChevronRight,
  Trash2,
} from 'lucide-react';
import { useTheme } from '../../contexts';

export interface SessionInfo {
  session_id: string;
  title: string;
  last_ts: number;
  message_count: number;
}

interface SidebarProps {
  isOpen: boolean;
  toggleSidebar: () => void;
  sessions: SessionInfo[];
  currentSessionId: string | null;
  onSelectSession: (id: string) => void;
  onNewChat: () => void;
  onDeleteSession?: (id: string) => void;
}

export function Sidebar({
  isOpen,
  toggleSidebar,
  sessions,
  currentSessionId,
  onSelectSession,
  onNewChat,
  onDeleteSession,
}: SidebarProps) {
  const navigate = useNavigate();
  const location = useLocation();
  const { isDark, toggleTheme } = useTheme();
  const [hoveredSession, setHoveredSession] = useState<string | null>(null);

  const isActive = useCallback(
    (path: string) => location.pathname.startsWith(path),
    [location.pathname],
  );

  return (
    <>
      {/* Mobile overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 bg-black/30 z-40 md:hidden"
          onClick={toggleSidebar}
        />
      )}

      <aside
        className={`
          fixed md:relative z-50 h-full flex flex-col
          bg-gray-50 dark:bg-gray-950 border-r border-gray-200 dark:border-gray-800
          transition-all duration-300 ease-in-out
          ${isOpen ? 'w-64' : 'w-0 md:w-16'}
          overflow-hidden
        `}
      >
        {/* Header */}
        <div className="h-14 flex items-center justify-between px-3 border-b border-gray-200 dark:border-gray-800 shrink-0">
          {isOpen && (
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-400 to-violet-600 flex items-center justify-center">
                <span className="text-white font-bold text-sm">IV</span>
              </div>
              <span className="font-bold text-gray-800 dark:text-gray-100 text-sm">InsightVault</span>
            </div>
          )}
          <button
            onClick={toggleSidebar}
            className="p-1.5 rounded-lg text-gray-500 hover:bg-gray-200 dark:hover:bg-gray-800 transition-colors"
          >
            {isOpen ? <ChevronLeft className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
          </button>
        </div>

        {/* New Chat Button */}
        <div className="p-2 shrink-0">
          <button
            onClick={onNewChat}
            className={`w-full flex items-center gap-2 px-3 py-2 rounded-xl text-sm font-medium
              bg-violet-500 text-white hover:bg-violet-600 shadow-sm transition-all
              ${!isOpen ? 'justify-center' : ''}
            `}
          >
            <MessageSquarePlus className="w-4 h-4 shrink-0" />
            {isOpen && <span>新对话</span>}
          </button>
        </div>

        {/* Navigation */}
        <nav className="px-2 space-y-1 shrink-0">
          <NavButton
            icon={<BookOpen className="w-4 h-4" />}
            label="知识库"
            isOpen={isOpen}
            active={isActive('/knowledge')}
            onClick={() => navigate('/knowledge')}
          />
          <NavButton
            icon={<Upload className="w-4 h-4" />}
            label="文件导入"
            isOpen={isOpen}
            active={isActive('/import')}
            onClick={() => navigate('/import')}
          />
          <NavButton
            icon={<BarChart3 className="w-4 h-4" />}
            label="性能与评测"
            isOpen={isOpen}
            active={isActive('/performance')}
            onClick={() => navigate('/performance')}
          />
        </nav>

        {/* Divider */}
        <div className="mx-3 my-2 border-t border-gray-200 dark:border-gray-800" />

        {/* Session List */}
        <div className="flex-1 overflow-y-auto px-2 space-y-0.5">
          {isOpen && sessions.length === 0 && (
            <p className="text-xs text-gray-400 text-center py-4">暂无对话</p>
          )}
          {isOpen &&
            sessions.map((s) => (
              <div
                key={s.session_id}
                className="relative"
                onMouseEnter={() => setHoveredSession(s.session_id)}
                onMouseLeave={() => setHoveredSession(null)}
              >
                <button
                  onClick={() => onSelectSession(s.session_id)}
                  className={`w-full text-left px-3 py-2 rounded-lg text-xs truncate transition-colors ${s.session_id === currentSessionId
                    ? 'bg-violet-100 dark:bg-violet-900/30 text-violet-700 dark:text-violet-300 font-medium'
                    : 'text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-800'
                    }`}
                  title={s.title}
                >
                  {s.title || s.session_id.slice(0, 20) + '...'}
                </button>
                {hoveredSession === s.session_id && onDeleteSession && (
                  <button
                    onClick={(e) => {
                      e.stopPropagation();
                      onDeleteSession(s.session_id);
                    }}
                    className="absolute right-1 top-1/2 -translate-y-1/2 p-1 rounded hover:bg-red-100 dark:hover:bg-red-900/30 text-gray-400 hover:text-red-500 transition-colors"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
            ))}
        </div>

        {/* Footer: Theme toggle */}
        <div className="p-2 border-t border-gray-200 dark:border-gray-800 shrink-0">
          <button
            onClick={toggleTheme}
            className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-800 transition-colors ${!isOpen ? 'justify-center' : ''}`}
          >
            {isDark ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
            {isOpen && <span>{isDark ? '浅色模式' : '深色模式'}</span>}
          </button>
        </div>
      </aside>
    </>
  );
}

function NavButton({
  icon,
  label,
  isOpen,
  active,
  onClick,
}: {
  icon: React.ReactNode;
  label: string;
  isOpen: boolean;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors ${active
        ? 'bg-violet-100 dark:bg-violet-900/30 text-violet-700 dark:text-violet-300 font-medium'
        : 'text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-800'
        } ${!isOpen ? 'justify-center' : ''}`}
      title={label}
    >
      {icon}
      {isOpen && <span>{label}</span>}
    </button>
  );
}
