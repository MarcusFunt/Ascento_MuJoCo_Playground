import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    proxy: {
      '/api': process.env.ASCENTO_DASHBOARD_DEV_API || 'http://127.0.0.1:8000',
    },
  },
})
