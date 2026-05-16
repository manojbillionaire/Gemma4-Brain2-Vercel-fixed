import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { defineConfig } from 'vite';

export default defineConfig({
  plugins: [react(), tailwindcss()],
  define: {},
  resolve: {
    alias: { '@': path.resolve(__dirname, '.') },
  },
  server: {
    headers: {
      // Required for wllama multi-thread (SharedArrayBuffer)
      'Cross-Origin-Opener-Policy':   'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
    hmr: process.env.DISABLE_HMR !== 'true',
  },
  preview: {
    headers: {
      'Cross-Origin-Opener-Policy':   'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  build: {
    outDir: 'dist',
    sourcemap: false,
    chunkSizeWarningLimit: 4000,
    rollupOptions: {
      output: {
        manualChunks: {
          react:    ['react', 'react-dom'],
          motion:   ['motion'],
          markdown: ['react-markdown'],
          docx:     ['docx', 'file-saver', 'jspdf'],
          wllama:   ['@wllama/wllama'],
        },
      },
    },
  },
  optimizeDeps: {
    exclude: ['@wllama/wllama'],  // ships its own ESM + WASM, skip pre-bundle
  },
  worker: { format: 'es' },
});
