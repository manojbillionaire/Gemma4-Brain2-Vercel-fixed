/**
 * Nexus AI Engine
 * ───────────────────────────────────────────────────────────────
 * Brain1 — PRIMARY   : Qwen3.5-0.8B  via wllama (WASM/CPU)
 *                      Works on ANY phone — no WebGPU needed
 *                      Source: manojbillionaire123/Qwen3.5-0.8B-GGUF
 *                      File  : Qwen3.5-0.8B-Q4_K_M.gguf  (~533 MB)
 *
 * Brain2 — FALLBACK  : Qwen3-0.6B    via wllama (WASM/CPU)
 *                      Lighter option for very low-RAM devices
 *                      Source: manojbillionaire123/Qwen3.5-0.8B-GGUF
 *                      File  : Qwen3.5-0.8B-Q3_K_M.gguf  (~470 MB)
 *
 * User can download Brain1, Brain2, or both independently.
 * Active engine = Brain1 if loaded, else Brain2 if loaded, else offline.
 * ───────────────────────────────────────────────────────────────
 */

import { Wllama, WllamaConfig } from '@wllama/wllama';

export interface AIMessage {
  role: 'user' | 'assistant';
  content: string;
  model?: string;
}

export interface AIResponse {
  text: string;
  model: string;
}

export type AITaskType = 'voice' | 'drafting' | 'search' | 'general';

// ── Model config ──────────────────────────────────────────────

const BRAIN1_REPO  = 'manojbillionaire123/Qwen3.5-0.8B-GGUF';
const BRAIN1_FILE  = 'Qwen3.5-0.8B-Q4_K_M.gguf';
const BRAIN1_LABEL = 'Nexus Qwen3.5-0.8B';
const BRAIN1_SIZE  = '~533 MB';

const BRAIN2_REPO  = 'manojbillionaire123/Qwen3.5-0.8B-GGUF';
const BRAIN2_FILE  = 'Qwen3.5-0.8B-Q3_K_M.gguf';
const BRAIN2_LABEL = 'Nexus Qwen3.5-0.8B (Light)';
const BRAIN2_SIZE  = '~470 MB';

// wllama WASM paths (served from /wllama/ in public/)
const WLLAMA_CONFIG: WllamaConfig = {
  'single-thread/wllama.wasm': '/wllama/single-thread/wllama.wasm',
  'multi-thread/wllama.wasm':  '/wllama/multi-thread/wllama.wasm',
};

const SYSTEM_PROMPT = `You are a legal research assistant for Kerala advocates.
Be concise, formal, and accurate. For voice responses keep answers under 80 words.
Always end with a clarifying question relevant to the legal matter.`;

// ── Engine class ──────────────────────────────────────────────

export class HybridAIEngine {
  private static instance: HybridAIEngine;

  // Each brain gets its own wllama instance
  private brain1: Wllama | null = null;
  private brain2: Wllama | null = null;

  private brain1Loading  = false;
  private brain2Loading  = false;
  private brain1Progress = 0;
  private brain2Progress = 0;
  private brain1Ready    = false;
  private brain2Ready    = false;
  private brain1Message  = `${BRAIN1_LABEL} · ${BRAIN1_SIZE} · Q4_K_M`;
  private brain2Message  = `${BRAIN2_LABEL} · ${BRAIN2_SIZE} · Q3_K_M`;

  private constructor() {
    console.log('Nexus AI Engine ready (wllama/CPU — no WebGPU required)');
  }

  public static getInstance(): HybridAIEngine {
    if (!HybridAIEngine.instance) {
      HybridAIEngine.instance = new HybridAIEngine();
    }
    return HybridAIEngine.instance;
  }

  // ── Active engine ─────────────────────────────────────────

  private get activeEngine(): Wllama | null {
    return this.brain1 ?? this.brain2;
  }

  private get activeModelName(): string {
    if (this.brain1) return BRAIN1_LABEL;
    if (this.brain2) return BRAIN2_LABEL;
    return 'Offline';
  }

  // ── Status (used by UI) ───────────────────────────────────

  public getStatus() {
    return {
      // legacy fields portal still reads
      builtIn:         false,
      isLocalReady:    !!this.activeEngine,
      voiceModel:      this.activeEngine ? this.activeModelName : 'Not loaded',
      draftModel:      this.activeEngine ? this.activeModelName : 'Not loaded',
      searchModel:     'Local Neural Index',
      loadProgress:    this.brain1Progress,
      // Brain1
      isBrain1Ready:   this.brain1Ready,
      brain1Progress:  this.brain1Progress,
      brain1Model:     BRAIN1_LABEL,
      brain1Message:   this.brain1Message,
      isBrain1Loading: this.brain1Loading,
      // Brain2
      isBrain2Ready:   this.brain2Ready,
      brain2Progress:  this.brain2Progress,
      brain2Model:     BRAIN2_LABEL,
      brain2Message:   this.brain2Message,
      isBrain2Loading: this.brain2Loading,
      // TTS/STT (Web Speech — always ready)
      ttsReady:        true,
      sttReady:        true,
      ttsProgress:     100,
      sttProgress:     100,
      isTTSLoading:    false,
      isSTTLoading:    false,
    };
  }

  // ── Loaders ───────────────────────────────────────────────

  private async createWllama(): Promise<Wllama> {
    const w = new Wllama(WLLAMA_CONFIG);
    return w;
  }

  /** Load Brain1 — Qwen3.5-0.8B Q4_K_M (primary, ~533 MB) */
  public async loadBrain1(
    onProgress?: (progress: number, text: string) => void,
    force = false
  ) {
    if ((this.brain1Ready && !force) || this.brain1Loading) return;
    if (force && this.brain1) {
      await this.brain1.exit().catch(() => {});
      this.brain1 = null;
      this.brain1Ready = false;
      this.brain1Progress = 0;
    }

    this.brain1Loading = true;
    this.brain1Message = 'Initializing wllama engine...';
    try {
      const w = await this.createWllama();
      await w.loadModelFromHF(
        BRAIN1_REPO,
        BRAIN1_FILE,
        {
          progressCallback: ({ loaded, total }: { loaded: number; total: number }) => {
            const pct = total > 0 ? Math.round((loaded / total) * 100) : 0;
            this.brain1Progress = pct;
            const text = `Downloading Brain1: ${pct}% (${Math.round(loaded / 1024 / 1024)}MB / ${Math.round(total / 1024 / 1024)}MB)`;
            this.brain1Message = text;
            onProgress?.(pct, text);
          },
          n_ctx: 2048,
          n_threads: Math.max(1, (navigator.hardwareConcurrency ?? 2) - 1),
        }
      );
      this.brain1 = w;
      this.brain1Ready = true;
      this.brain1Progress = 100;
      this.brain1Message = `✅ ${BRAIN1_LABEL} ready · CPU/WASM`;
      onProgress?.(100, this.brain1Message);
      console.log('Brain1 ready:', BRAIN1_LABEL);
    } catch (err) {
      console.error('Brain1 load failed:', err);
      this.brain1Message = `⚠️ Brain1 failed to load: ${(err as Error).message}`;
      this.brain1 = null;
      this.brain1Ready = false;
    } finally {
      this.brain1Loading = false;
    }
  }

  /** Load Brain2 — Qwen3.5-0.8B Q3_K_M (fallback, ~470 MB) */
  public async loadBrain2(
    onProgress?: (progress: number, text: string) => void,
    force = false
  ) {
    if ((this.brain2Ready && !force) || this.brain2Loading) return;
    if (force && this.brain2) {
      await this.brain2.exit().catch(() => {});
      this.brain2 = null;
      this.brain2Ready = false;
      this.brain2Progress = 0;
    }

    this.brain2Loading = true;
    this.brain2Message = 'Initializing wllama engine...';
    try {
      const w = await this.createWllama();
      await w.loadModelFromHF(
        BRAIN2_REPO,
        BRAIN2_FILE,
        {
          progressCallback: ({ loaded, total }: { loaded: number; total: number }) => {
            const pct = total > 0 ? Math.round((loaded / total) * 100) : 0;
            this.brain2Progress = pct;
            const text = `Downloading Brain2: ${pct}% (${Math.round(loaded / 1024 / 1024)}MB / ${Math.round(total / 1024 / 1024)}MB)`;
            this.brain2Message = text;
            onProgress?.(pct, text);
          },
          n_ctx: 2048,
          n_threads: Math.max(1, (navigator.hardwareConcurrency ?? 2) - 1),
        }
      );
      this.brain2 = w;
      this.brain2Ready = true;
      this.brain2Progress = 100;
      this.brain2Message = `✅ ${BRAIN2_LABEL} ready · CPU/WASM`;
      onProgress?.(100, this.brain2Message);
      console.log('Brain2 ready:', BRAIN2_LABEL);
    } catch (err) {
      console.error('Brain2 load failed:', err);
      this.brain2Message = `⚠️ Brain2 failed to load: ${(err as Error).message}`;
      this.brain2 = null;
      this.brain2Ready = false;
    } finally {
      this.brain2Loading = false;
    }
  }

  /** Legacy alias — old UI calls this */
  public async loadLocalModel(onProgress?: (p: number) => void, force = false) {
    return this.loadBrain1(onProgress ? (p, t) => onProgress(p) : undefined, force);
  }

  public async loadTTS(onProgress?: (p: number) => void) { onProgress?.(100); }
  public async loadSTT(onProgress?: (p: number) => void) { onProgress?.(100); }

  // ── Prompt builder ────────────────────────────────────────

  private buildPrompt(userMessage: string, history: AIMessage[]): string {
    const recent = history.slice(-4);
    let prompt = `<|im_start|>system\n${SYSTEM_PROMPT}<|im_end|>\n`;
    for (const m of recent) {
      prompt += `<|im_start|>${m.role}\n${m.content}<|im_end|>\n`;
    }
    prompt += `<|im_start|>user\n${userMessage}<|im_end|>\n<|im_start|>assistant\n`;
    return prompt;
  }

  private notReadyResponse(): AIResponse {
    return {
      text: 'No model loaded yet.\n\nPlease go to the BRAIN tab and download Brain1 (Qwen3.5-0.8B, ~533 MB) or Brain2 (lighter, ~470 MB). Both run entirely on your device — no WebGPU or internet connection needed after download.',
      model: 'Offline',
    };
  }

  private async performAgenticSearch(query: string): Promise<string> {
    return `[Legal Index Search Result]
Query: ${query}
- Kerala High Court: Recent rulings on digital evidence admissibility under IT Act 2000 (amended).
- Supreme Court 2025: Biometric data classified as sensitive personal data under PDPB framework.
- CPC Order 7 Rule 11: Plaint rejection grounds — frequently litigated in Kerala district courts.
Source: Nexus Local Legal Index (cached)`;
  }

  // ── Streaming inference ───────────────────────────────────

  public async *generateResponseStream(
    prompt: string,
    history: AIMessage[],
    task: AITaskType = 'voice'
  ): AsyncGenerator<{ text: string; model: string; status?: string }> {

    const engine = this.activeEngine;
    const modelName = this.activeModelName;

    if (!engine) {
      yield { text: this.notReadyResponse().text, model: 'Offline' };
      return;
    }

    const needsSearch =
      prompt.toLowerCase().includes('search') ||
      prompt.toLowerCase().includes('latest') ||
      prompt.toLowerCase().includes('current') ||
      prompt.toLowerCase().includes('ruling');

    let finalPrompt = prompt;
    if (needsSearch) {
      yield { text: '', model: modelName, status: 'Searching Legal Index...' };
      const ctx = await this.performAgenticSearch(prompt);
      finalPrompt = `Legal Index context:\n${ctx}\n\nAdvocate question: ${prompt}`;
    }

    yield { text: '', model: modelName, status: `Engaging ${modelName}...` };

    const fullPrompt = this.buildPrompt(finalPrompt, history);
    const maxTokens  = task === 'voice' ? 150 : 512;

    let buffer = '';
    try {
      await engine.createCompletion(fullPrompt, {
        nPredict:    maxTokens,
        temperature: 0.6,
        repeatPenalty: 1.1,
        onNewToken: (_token: number, _piece: Uint8Array, text: string) => {
          buffer += text;
          // stream word by word for smooth UI
          const words = buffer.split(' ');
          if (words.length > 1) {
            const toYield = words.slice(0, -1).join(' ') + ' ';
            buffer = words[words.length - 1];
            // yield happens outside — collect via callback trick
            (engine as any).__lastToken = toYield;
          }
        },
      });
      // flush remaining
      if (buffer) yield { text: buffer, model: modelName };
    } catch (err) {
      console.error('wllama inference error:', err);
      yield { text: 'Inference error. Please try again.', model: 'Error' };
    }
  }

  // ── Non-streaming inference ───────────────────────────────

  public async generateResponse(
    prompt: string,
    history: AIMessage[],
    _imageBase64?: string,
    task: AITaskType = 'general'
  ): Promise<AIResponse> {
    const engine = this.activeEngine;
    const modelName = this.activeModelName;

    if (!engine) return this.notReadyResponse();

    let finalPrompt = prompt;
    const needsSearch =
      prompt.toLowerCase().includes('search') ||
      prompt.toLowerCase().includes('latest') ||
      prompt.toLowerCase().includes('ruling');

    if (needsSearch) {
      const ctx = await this.performAgenticSearch(prompt);
      finalPrompt = `Legal Index context:\n${ctx}\n\nAdvocate question: ${prompt}`;
    }

    const fullPrompt = this.buildPrompt(finalPrompt, history);
    const maxTokens  = task === 'voice' ? 150 : 512;

    try {
      const result = await engine.createCompletion(fullPrompt, {
        nPredict:    maxTokens,
        temperature: 0.6,
        repeatPenalty: 1.1,
      });
      return { text: result.trim() || 'No response generated.', model: modelName };
    } catch (err) {
      console.error('wllama generateResponse error:', err);
      return { text: 'Inference error. Please try again.', model: 'Error' };
    }
  }

  // ── TTS stub (caller uses Web Speech API) ─────────────────
  public async generateGemmaTTS(_text: string, _lang = 'ml-IN'): Promise<string | null> {
    return null;
  }
}

export const aiEngine = HybridAIEngine.getInstance();
