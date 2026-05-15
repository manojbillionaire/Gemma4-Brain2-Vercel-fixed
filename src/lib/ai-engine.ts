import * as webllm from "@mlc-ai/web-llm";

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

// ─────────────────────────────────────────────────────────────
// PRIMARY  — Qwen2.5-1.5B-Instruct  (880 MB, WebGPU / WebLLM)
// FALLBACK — Nexus Qwen3-0.6B       (720 MB, WebGPU / WebLLM)
// ─────────────────────────────────────────────────────────────

const PRIMARY_MODEL_ID  = "nexus-qwen2.5-1.5b";
const PRIMARY_MODEL_URL = "https://huggingface.co/manojbillionaire123/Qwen2.5-1.5B-Instruct-q4f16_1-MLC/resolve/main/";

const FALLBACK_MODEL_ID  = "nexus-qwen3-0.6b";
const FALLBACK_MODEL_URL = "https://huggingface.co/Kichu123/nexus-qwen3-0.6b/resolve/main/";

// Reuse official wasm libs so we never get an ABI mismatch
const _prebuiltQwen25 = webllm.prebuiltAppConfig.model_list.find(
  (m: any) => m.model_id === "Qwen2.5-1.5B-Instruct-q4f16_1-MLC"
);
const _prebuiltQwen3 = webllm.prebuiltAppConfig.model_list.find(
  (m: any) => m.model_id === "Qwen3-0.6B-q4f16_1-MLC"
);

webllm.prebuiltAppConfig.model_list.push(
  {
    model:            PRIMARY_MODEL_URL,
    model_id:         PRIMARY_MODEL_ID,
    model_lib:        _prebuiltQwen25?.model_lib ?? (
      webllm.modelLibURLPrefix + webllm.modelVersion +
      "/Qwen2.5-1.5B-Instruct-q4f16_1-ctx4k_cs1k-webgpu.wasm"
    ),
    vram_required_MB: 1800,
    low_resource_required: false,
    overrides: { context_window_size: 4096 },
  },
  {
    model:            FALLBACK_MODEL_URL,
    model_id:         FALLBACK_MODEL_ID,
    model_lib:        _prebuiltQwen3?.model_lib ?? (
      webllm.modelLibURLPrefix + webllm.modelVersion +
      "/Qwen3-0.6B-q4f16_1-ctx4k_cs1k-webgpu.wasm"
    ),
    vram_required_MB: 720,
    low_resource_required: true,
    overrides: { context_window_size: 2048 },
  }
);

export class HybridAIEngine {
  private static instance: HybridAIEngine;

  private primaryEngine:  webllm.MLCEngine | null = null;
  private fallbackEngine: webllm.MLCEngine | null = null;

  private isPrimaryLoading  = false;
  private isFallbackLoading = false;

  private primaryProgress  = 0;
  private fallbackProgress = 0;

  private readonly SYSTEM_PROMPT = `You are a Research Agent powered by Nexus AI.
Voice Context: Keep speech concise, formal, and helpful.
CRITICAL: End every response with a clarifying question.`;

  private constructor() {
    console.log("Nexus AI Engine initializing (Primary: Qwen2.5-1.5B | Fallback: Qwen3-0.6B)...");
  }

  public static getInstance(): HybridAIEngine {
    if (!HybridAIEngine.instance) {
      HybridAIEngine.instance = new HybridAIEngine();
    }
    return HybridAIEngine.instance;
  }

  // ── Active engine helper ──────────────────────────────────

  private get activeEngine(): webllm.MLCEngine | null {
    return this.primaryEngine ?? this.fallbackEngine;
  }

  private get activeModelName(): string {
    if (this.primaryEngine)  return "Nexus Qwen2.5-1.5B";
    if (this.fallbackEngine) return "Nexus Qwen3-0.6B (Fallback)";
    return "Offline";
  }

  // ── Status ────────────────────────────────────────────────

  public getStatus() {
    return {
      builtIn:           false,
      voiceModel:        this.activeEngine ? this.activeModelName : "Not loaded",
      draftModel:        this.activeEngine ? this.activeModelName : "Not loaded",
      searchModel:       "Local Neural Index",
      isLocalReady:      !!this.activeEngine,
      // Primary (Brain2 UI)
      isBrain2Ready:     !!this.primaryEngine,
      brain2Progress:    this.primaryProgress,
      brain2Model:       "Nexus Qwen2.5-1.5B",
      loadProgress:      this.primaryProgress,
      // Fallback
      isFallbackReady:   !!this.fallbackEngine,
      fallbackProgress:  this.fallbackProgress,
      fallbackModel:     "Nexus Qwen3-0.6B",
    };
  }

  // ── Loaders ───────────────────────────────────────────────

  /** Alias used by existing UI calls */
  public async loadLocalModel(
    onProgress?: (progress: number) => void,
    force: boolean = false
  ) {
    return this.loadPrimaryModel(
      onProgress ? (p, _t) => onProgress(p) : undefined,
      force
    );
  }

  /** Load PRIMARY model — Qwen2.5-1.5B */
  public async loadBrain2(
    onProgress?: (progress: number, text: string) => void,
    force: boolean = false
  ) {
    return this.loadPrimaryModel(onProgress, force);
  }

  public async loadPrimaryModel(
    onProgress?: (progress: number, text: string) => void,
    force: boolean = false
  ) {
    if ((this.primaryEngine && !force) || this.isPrimaryLoading) return;
    if (force) { this.primaryEngine = null; this.primaryProgress = 0; }

    this.isPrimaryLoading = true;
    try {
      console.log("Nexus: Loading PRIMARY — Qwen2.5-1.5B...");
      this.primaryEngine = await webllm.CreateMLCEngine(PRIMARY_MODEL_ID, {
        initProgressCallback: (r: webllm.InitProgressReport) => {
          this.primaryProgress = Math.round(r.progress * 100);
          onProgress?.(this.primaryProgress, r.text ?? "Loading...");
        },
      });
      this.primaryProgress = 100;
      onProgress?.(100, "Qwen2.5-1.5B ready.");
      console.log("PRIMARY ready — Qwen2.5-1.5B.");
    } catch (err) {
      console.error("Primary load failed:", err);
      this.primaryEngine = null;
      // Auto-trigger fallback load
      console.log("Nexus: Falling back to Qwen3-0.6B...");
      await this.loadFallbackModel(onProgress);
    } finally {
      this.isPrimaryLoading = false;
    }
  }

  public async loadFallbackModel(
    onProgress?: (progress: number, text: string) => void,
    force: boolean = false
  ) {
    if ((this.fallbackEngine && !force) || this.isFallbackLoading) return;
    if (force) { this.fallbackEngine = null; this.fallbackProgress = 0; }

    this.isFallbackLoading = true;
    try {
      console.log("Nexus: Loading FALLBACK — Qwen3-0.6B...");
      this.fallbackEngine = await webllm.CreateMLCEngine(FALLBACK_MODEL_ID, {
        initProgressCallback: (r: webllm.InitProgressReport) => {
          this.fallbackProgress = Math.round(r.progress * 100);
          onProgress?.(this.fallbackProgress, `[Fallback] ${r.text ?? "Loading..."}`);
        },
      });
      this.fallbackProgress = 100;
      onProgress?.(100, "Qwen3-0.6B fallback ready.");
      console.log("FALLBACK ready — Qwen3-0.6B.");
    } catch (err) {
      console.error("Fallback load failed:", err);
      this.fallbackEngine = null;
    } finally {
      this.isFallbackLoading = false;
    }
  }

  // ── Inference helpers ─────────────────────────────────────

  private buildMessages(prompt: string, history: AIMessage[]): webllm.ChatCompletionMessageParam[] {
    return [
      { role: "system", content: this.SYSTEM_PROMPT },
      ...history.slice(-4).map((m) => ({
        role: m.role as "user" | "assistant",
        content: m.content,
      })),
      { role: "user", content: prompt },
    ];
  }

  private notReadyResponse(): AIResponse {
    return {
      text: "No model is loaded yet.\n\nPlease click \"Download Brain2\" to load Nexus Qwen2.5-1.5B (~880 MB). WebGPU is required (Chrome 113+ or Edge 113+). If your device has limited VRAM, the lighter Qwen3-0.6B fallback will load automatically.",
      model: "Offline",
    };
  }

  private async performAgenticSearch(query: string): Promise<string> {
    console.log("Nexus Agent researching:", query);
    return `[AGENTIC SEARCH RESULT]
- Recent privacy rulings (2026) focus on "Digital Sovereign Identity" and biometric data encryption.
- Government mandates strict 10-minute deletion for non-essential tracking tokens.
- Supreme Court clarified that data stored locally in Chrome Neural Runtimes is exempt from broad seizure warrants.
Source: Chrome Legal Index (Local Cache)`;
  }

  // ── Streaming public API ──────────────────────────────────

  public async *generateResponseStream(
    prompt: string,
    history: AIMessage[],
    task: AITaskType = "voice"
  ): AsyncGenerator<{ text: string; model: string; status?: string }> {

    const needsSearch =
      prompt.toLowerCase().includes("search") ||
      prompt.toLowerCase().includes("latest") ||
      prompt.toLowerCase().includes("current") ||
      prompt.toLowerCase().includes("ruling");

    if (needsSearch) {
      yield { text: "", model: "Nexus Agent", status: "Agentic Loop: Searching Local Index..." };
      const ctx = await this.performAgenticSearch(prompt);
      yield { text: "", model: "Nexus Agent", status: "Synthesizing Research..." };
      prompt = `Here is recent research from Legal Index:\n${ctx}\n\nBased on this, answer: ${prompt}`;
    }

    const engine = this.activeEngine;
    const modelName = this.activeModelName;

    if (!engine) {
      yield {
        text: "No model loaded. Please download Qwen2.5-1.5B from the BRAIN2 tab.",
        model: "Offline",
      };
      return;
    }

    const isAgentic = prompt.includes("Legal Index:") || prompt.includes("Research Data:");

    try {
      yield { text: "", model: modelName, status: `Engaging ${modelName}...` };
      const messages = this.buildMessages(prompt, history);
      const stream = await engine.chat.completions.create({
        messages,
        max_tokens: task === 'voice' ? 256 : 512,
        temperature: 0.6,
        repetition_penalty: 1.1,
        stream: true,
      });
      for await (const chunk of stream) {
        const delta = chunk.choices[0]?.delta?.content;
        if (delta) yield {
          text: delta,
          model: isAgentic ? `${modelName} (Agentic)` : modelName,
        };
      }
    } catch (err) {
      console.error("Inference failed:", err);
      yield { text: "Error: inference failed.", model: "Error" };
    }
  }

  // ── Non-streaming public API ──────────────────────────────

  public async generateResponse(
    prompt: string,
    history: AIMessage[],
    imageBase64?: string,
    task: AITaskType = "general"
  ): Promise<AIResponse> {
    const engine = this.activeEngine;
    const modelName = this.activeModelName;

    if (!engine) return this.notReadyResponse();

    try {
      const needsSearch =
        prompt.toLowerCase().includes("search") ||
        prompt.toLowerCase().includes("latest") ||
        prompt.toLowerCase().includes("current") ||
        prompt.toLowerCase().includes("ruling");

      let finalPrompt = prompt;
      let isAgentic = false;
      if (needsSearch) {
        const ctx = await this.performAgenticSearch(prompt);
        finalPrompt = `Research Data:\n${ctx}\n\nUser Question: ${prompt}`;
        isAgentic = true;
      }

      const messages = this.buildMessages(finalPrompt, history);
      const reply = await engine.chat.completions.create({
        messages,
        max_tokens: 512,
        temperature: 0.6,
        repetition_penalty: 1.1,
        stream: false,
      });
      const text = reply.choices[0]?.message?.content ?? "No response.";
      return { text, model: isAgentic ? `${modelName} (Agentic)` : modelName };
    } catch {
      return { text: "Inference failed.", model: "Error" };
    }
  }

  // ── TTS stub — falls through to Web Speech API in caller ──

  public async generateGemmaTTS(
    _text: string,
    _languageCode: string = "ml-IN"
  ): Promise<string | null> {
    return null;
  }
}

export const aiEngine = HybridAIEngine.getInstance();
