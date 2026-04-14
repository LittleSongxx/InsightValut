import { useState, useEffect, useCallback } from 'react';
import type { ReactNode } from 'react';
import { ThemeContext } from './themeContext';

export function ThemeProvider({ children }: { children: ReactNode }) {
  const [isDark, setIsDark] = useState(() => {
    const stored = localStorage.getItem('theme');
    if (stored) return stored === 'dark';
    return window.matchMedia('(prefers-color-scheme: dark)').matches;
  });

  useEffect(() => {
    const root = document.documentElement;
    root.classList.add('theme-transition');
    root.classList.toggle('dark', isDark);
    const cleanupTimer = window.setTimeout(() => {
      root.classList.remove('theme-transition');
    }, 260);
    localStorage.setItem('theme', isDark ? 'dark' : 'light');
    return () => {
      window.clearTimeout(cleanupTimer);
      root.classList.remove('theme-transition');
    };
  }, [isDark]);

  const toggleTheme = useCallback(() => setIsDark((d) => !d), []);

  return (
    <ThemeContext.Provider value={{ isDark, toggleTheme }}>
      {children}
    </ThemeContext.Provider>
  );
}


