import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  const queryTarget = env.VITE_QUERY_TARGET || 'http://localhost:8001'
  const importTarget = env.VITE_IMPORT_TARGET || 'http://localhost:18000'

  return {
    plugins: [react(), tailwindcss()],
    cacheDir: 'node_modules/.vite',
    build: {
      rollupOptions: {
        output: {
          manualChunks(id) {
            if (!id.includes('node_modules')) return undefined
            if (id.includes('react-router')) return 'router'
            if (id.includes('recharts')) return 'charts'
            if (id.includes('react-markdown') || id.includes('remark-gfm') || id.includes('rehype-highlight')) return 'markdown'
            if (id.includes('react') || id.includes('scheduler')) return 'react-vendor'
            return 'vendor'
          },
        },
      },
    },
    server: {
      port: 3002,
      proxy: {
        '/api/query': {
          target: queryTarget,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api\/query/, ''),
        },
        '/api/import': {
          target: importTarget,
          changeOrigin: true,
          rewrite: (path) => path.replace(/^\/api\/import/, ''),
        },
      },
    },
  }
})
