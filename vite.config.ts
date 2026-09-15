import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    tailwindcss(),
    react({
      babel: {
        plugins: [['babel-plugin-react-compiler']],
      },
    }),
  ],
  resolve: {
    alias: {
      '@': path.resolve(path.dirname(fileURLToPath(import.meta.url)), 'src'),
    },
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/fpl-api': {
        target: 'https://fantasy.premierleague.com',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/fpl-api/, '/api'),
      },
    },
  },
  // `vite preview` does not use `server.proxy` unless mirrored here
  preview: {
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
      '/fpl-api': {
        target: 'https://fantasy.premierleague.com',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/fpl-api/, '/api'),
      },
    },
  },
})
