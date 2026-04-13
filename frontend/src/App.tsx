import { useState, useCallback, lazy, Suspense } from 'react';
import { Navigate, Route, Routes, useNavigate } from 'react-router-dom';
import { Menu } from 'lucide-react';
import { Sidebar } from './components';
import { clearHistory } from './services/api';
import { useSession } from './contexts';

const ChatPage = lazy(() => import('./pages/ChatPage'));
const KnowledgePage = lazy(() => import('./pages/KnowledgePage'));
const ImportPage = lazy(() => import('./pages/ImportPage'));
const PerformancePage = lazy(() => import('./pages/PerformancePage'));

function RouteFallback() {
  return (
    <div className="flex h-full min-h-[240px] items-center justify-center text-sm text-gray-500 dark:text-gray-400">
      加载中...
    </div>
  );
}

function MainLayout({ children }: { children: React.ReactNode }) {
  const navigate = useNavigate();
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const { sessions, currentSessionId, createNewSession, selectSession, refreshSessions } = useSession();

  const handleNewChat = useCallback(() => {
    createNewSession();
    navigate('/chat');
  }, [createNewSession, navigate]);

  const handleSelectSession = useCallback(
    (id: string) => {
      selectSession(id);
      navigate('/chat');
    },
    [navigate, selectSession],
  );

  const handleDeleteSession = useCallback(
    async (id: string) => {
      try {
        await clearHistory(id);
        if (currentSessionId === id) {
          selectSession(null);
          navigate('/chat');
        }
        await refreshSessions();
      } catch {
        return;
      }
    },
    [currentSessionId, navigate, refreshSessions, selectSession],
  );

  return (
    <div className="flex h-screen bg-white dark:bg-gray-900 text-gray-900 dark:text-gray-100 transition-colors duration-200">
      <Sidebar
        isOpen={isSidebarOpen}
        toggleSidebar={() => setIsSidebarOpen(!isSidebarOpen)}
        sessions={sessions}
        currentSessionId={currentSessionId}
        onSelectSession={handleSelectSession}
        onNewChat={handleNewChat}
        onDeleteSession={handleDeleteSession}
      />

      <div className="flex-1 flex flex-col h-full relative">
        <header className="h-14 border-b border-gray-200 dark:border-gray-800 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm flex items-center px-4 justify-between z-50">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setIsSidebarOpen(!isSidebarOpen)}
              className="p-2 -ml-2 text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors md:hidden"
            >
              <Menu className="w-5 h-5" />
            </button>
          </div>
          <div className="flex items-center gap-2 text-xs text-gray-500">
            {currentSessionId && (
              <span className="hidden sm:inline font-mono bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded truncate max-w-[200px]">
                {currentSessionId}
              </span>
            )}
          </div>
        </header>

        <main className="flex-1 flex flex-col overflow-hidden relative bg-gradient-to-b from-white to-gray-50 dark:from-gray-900 dark:to-gray-950">
          {children}
        </main>
      </div>
    </div>
  );
}

function App() {
  return (
    <Suspense fallback={<RouteFallback />}>
      <Routes>
        <Route
          path="/chat"
          element={
            <MainLayout>
              <ChatPage />
            </MainLayout>
          }
        />
        <Route
          path="/knowledge"
          element={
            <MainLayout>
              <KnowledgePage />
            </MainLayout>
          }
        />
        <Route
          path="/import"
          element={
            <MainLayout>
              <ImportPage />
            </MainLayout>
          }
        />
        <Route
          path="/performance"
          element={
            <MainLayout>
              <PerformancePage />
            </MainLayout>
          }
        />
        <Route path="/" element={<Navigate to="/chat" replace />} />
        <Route path="*" element={<Navigate to="/chat" replace />} />
      </Routes>
    </Suspense>
  );
}

export default App;
