import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      // Any /api/* request from the frontend is forwarded to FastAPI on :8000
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        rewrite: (path) => path,  // keep /api prefix (FastAPI uses /api/chat etc.)
      },
    },
  },
})
