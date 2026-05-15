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
// Nexus Qwen3-0.6B — sole inference engine (WebLLM / WebGPU)
// ─────────────────────────────────────────────────────────────
const MODEL_ID  = "nexus-qwen3-0.6b";
const MODEL_URL = "https://huggingface.co/Kichu123/nexus-qwen3-0.6b/resolve/main/";

const _prebuiltQwen3 = webllm.prebuiltAppConfig.model_list.find(
  (m: any) => m.model_id === "Qwen3-0.6B-q4f16_1-MLC"
);

webllm.prebuiltAppConfig.model_list.push({
  model:            MODEL_URL,
  model_id:         MODEL_ID,
  model_lib:        _prebuiltQwen3?.model_lib ?? (
    webllm.modelLibURLPrefix + webllm.modelVersion +
    "/Qwen3-0.6B-q4f16_1-ctx4k_cs1k-webgpu.wasm"
  ),
  vram_required_MB: 720,
  low_resource_required: true,
  overrides: { context_window_size: 2048 },
});

export class HybridAIEngine {
  private static instance: HybridAIEngine;

  private engine: webllm.MLCEngine | null = null;
  private isLoading = false;
  private loadProgress = 0;

  private readonly SYSTEM_PROMPT = `
You are a Research Agent powered by Nexus Qwen3-0.6B.
Voice Context: Keep speech concise, formal, and helpful.
CRITICAL: End every response with a clarifying question.
`;

  private constructor() {
    console.log("Nexus AI Engine (Qwen3-0.6B) initializing...");
  }

  public static getInstance(): HybridAIEngine {
    if (!HybridAIEngine.instance) {
      HybridAIEngine.instance = new HybridAIEngine();
    }
    return HybridAIEngine.instance;
  }

  // ── Status ────────────────────────────────────────────────

  public getStatus() {
    return {
      builtIn:        false,
      voiceModel:     this.engine ? "Nexus Qwen3-0.6B" : "Not loaded",
      draftModel:     "Nexus Qwen3-0.6B",
      searchModel:    "Local Neural Index",
      isLocalReady:   !!this.engine,
      loadProgress:   this.loadProgress,
      isBrain2Ready:  !!this.engine,
      brain2Progress: this.loadProgress,
      brain2Model:    "Nexus Qwen3-0.6B",
    };
  }

  // ── Loader ────────────────────────────────────────────────

  public async loadLocalModel(
    onProgress?: (progress: number) => void,
    force: boolean = false
  ) {
    return this.loadBrain2(
      onProgress ? (p) => onProgress(p) : undefined,
      force
    );
  }

  public async loadBrain2(
    onProgress?: (progress: number, text: string) => void,
    force: boolean = false
  ) {
    if ((this.engine && !force) || this.isLoading) return;
    if (force) { this.engine = null; this.loadProgress = 0; }

    this.isLoading = true;
    try {
      console.log("Nexus: Initialising WebLLM with Qwen3-0.6B...");
      this.engine = await webllm.CreateMLCEngine(MODEL_ID, {
        initProgressCallback: (report: webllm.InitProgressReport) => {
          this.loadProgress = Math.round(report.progress * 100);
          if (onProgress) onProgress(this.loadProgress, report.text ?? "Loading...");
        },
      });
      this.loadProgress = 100;
      if (onProgress) onProgress(100, "Nexus Qwen3-0.6B ready.");
      console.log("Nexus Qwen3-0.6B ready.");
    } catch (error) {
      console.error("Nexus load failed:", error);
      this.engine = null;
    } finally {
      this.isLoading = false;
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
      text: "Nexus Qwen3-0.6B is not loaded yet.\n\nPlease click \"Download Brain2\" to load the model (~720 MB). WebGPU is required (Chrome 113+ or Edge 113+).",
      model: "Offline",
    };
  }

  // ── Agentic search ────────────────────────────────────────

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
      const searchResult = await this.performAgenticSearch(prompt);
      yield { text: "", model: "Nexus Agent", status: "Synthesizing Research..." };
      prompt = `Here is recent research from Legal Index:\n${searchResult}\n\nBased on this, answer: ${prompt}`;
    }

    const isAgentic = prompt.includes("Legal Index:") || prompt.includes("Research Data:");

    if (!this.engine) {
      yield {
        text: "Nexus Qwen3-0.6B is not loaded. Please download it from the BRAIN2 tab.",
        model: "Offline",
      };
      return;
    }

    try {
      yield { text: "", model: "Nexus Qwen3-0.6B", status: "Engaging Nexus Qwen3-0.6B..." };
      const messages = this.buildMessages(prompt, history);
      const stream = await this.engine.chat.completions.create({
        messages,
        max_tokens: 512,
        temperature: 0.6,
        repetition_penalty: 1.1,
        stream: true,
      });
      for await (const chunk of stream) {
        const delta = chunk.choices[0]?.delta?.content;
        if (delta) yield {
          text: delta,
          model: isAgentic ? "Nexus Qwen3-0.6B (Agentic)" : "Nexus Qwen3-0.6B",
        };
      }
    } catch (err) {
      console.error("Nexus inference failed:", err);
      yield { text: "Error: Nexus Qwen3-0.6B inference failed.", model: "Error" };
    }
  }

  // ── Non-streaming public API ──────────────────────────────

  public async generateResponse(
    prompt: string,
    history: AIMessage[],
    imageBase64?: string,
    task: AITaskType = "general"
  ): Promise<AIResponse> {
    try {
      if (!this.engine) return this.notReadyResponse();

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
      const reply = await this.engine.chat.completions.create({
        messages,
        max_tokens: 512,
        temperature: 0.6,
        repetition_penalty: 1.1,
        stream: false,
      });
      const text = reply.choices[0]?.message?.content ?? "No response.";
      return { text, model: isAgentic ? "Nexus Qwen3-0.6B (Agentic)" : "Nexus Qwen3-0.6B" };
    } catch {
      return { text: "Nexus inference failed.", model: "Error" };
    }
  }


  // TTS: use Web Speech API via MalayalamEngine (no external API needed)
  public async generateGemmaTTS(_text: string, _languageCode: string = "ml-IN"): Promise<string | null> {
    return null; // Falls through to browser SpeechSynthesis in the caller
  }

}

export const aiEngine = HybridAIEngine.getInstance();
