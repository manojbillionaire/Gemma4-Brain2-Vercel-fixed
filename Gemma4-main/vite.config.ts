import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import path from 'path';
import { defineConfig, loadEnv } from 'vite';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', '');
  return {
    plugins: [react(), tailwindcss()],
    define: {

    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.'),
      },
    },
    // Required for @mlc-ai/web-llm (SharedArrayBuffer / Atomics)
    server: {
      headers: {
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
      },
      hmr: process.env.DISABLE_HMR !== 'true',
    },
    preview: {
      headers: {
        'Cross-Origin-Opener-Policy': 'same-origin',
        'Cross-Origin-Embedder-Policy': 'require-corp',
      },
    },
    build: {
      outDir: 'dist',
      sourcemap: false,
      // WebLLM runtime is intentionally large (~6MB); raise the warning threshold
      chunkSizeWarningLimit: 7000,
      rollupOptions: {
        output: {
          // Chunk large deps so Vercel's 250 kB chunk warning stays quiet
          manualChunks: {
            react: ['react', 'react-dom'],
            motion: ['motion'],
            markdown: ['react-markdown'],
            docx: ['docx', 'file-saver', 'jspdf'],
            webllm: ['@mlc-ai/web-llm'],
          },
        },
      },
    },
    optimizeDeps: {
      exclude: ['@mlc-ai/web-llm'],   // WebLLM ships its own ESM, don't pre-bundle
    },
    worker: {
      format: 'es',
    },
  };
});
