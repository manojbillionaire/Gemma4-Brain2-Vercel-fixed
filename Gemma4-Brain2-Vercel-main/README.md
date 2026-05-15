# Nexus Justice — Gemma 4 Edition

AI-powered legal assistant for Kerala advocates. Runs entirely in the browser with three inference tiers:

| Priority | Engine | Model |
|----------|--------|-------|
| 1 | Chrome Built-in AI | Gemma 4 Nano (native) |
| 2 | Transformers.js (Brain1) | Qwen1.5-0.5B |
| 3 | WebLLM / WebGPU (Brain2) | Nexus Qwen3-0.6B |

---

## Deploy to Vercel (one-click)

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new)

### Steps

1. Push this repo to GitHub / GitLab / Bitbucket
2. Import the repo in [vercel.com/new](https://vercel.com/new)
3. Vercel auto-detects Vite — no framework config needed
4. Add environment variables (optional):

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Gemini API key for AI features |
| `VITE_SARVAM_API_KEY` | Sarvam AI Malayalam TTS |

5. Click **Deploy**

---

## Local development

```bash
cp .env.example .env.local   # add your keys
npm install
npm run dev                  # http://localhost:3000
```

## Build

```bash
npm run build    # outputs to dist/
npm run preview  # preview the dist build locally
```

---

## Notes

- **Brain2 (WebLLM)** requires Chrome 113+ or Edge 113+ with WebGPU enabled
- **SharedArrayBuffer** is enabled via `Cross-Origin-Opener-Policy: same-origin` + `Cross-Origin-Embedder-Policy: require-corp` headers — already configured in `vercel.json` and `index.html`
- Brain2 downloads ~350 MB on first use; subsequent loads use IndexedDB cache
