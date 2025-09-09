import { defineConfig, loadEnv } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, process.cwd(), '')
  
  return {
    plugins: [react()],
    base: mode === 'production' ? '/AutoStockAnalysis/' : '/',
    server: {
      port: 5173,
      proxy: {
        '/api': {
          target: env.VITE_API_BASE_URL || 'http://127.0.0.1:8000',
          changeOrigin: true,
        },
      },
    },
    build: {
      outDir: 'dist',
      sourcemap: false,
      rollupOptions: {
        output: {
          manualChunks: {
            vendor: ['react', 'react-dom'],
            charts: ['echarts', 'echarts-for-react']
          }
        }
      }
    },
    define: {
      __API_BASE_URL__: JSON.stringify(env.VITE_API_BASE_URL || 'http://localhost:8000'),
    },
  }
})
