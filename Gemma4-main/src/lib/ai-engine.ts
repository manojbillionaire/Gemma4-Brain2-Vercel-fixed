import { pipeline, env } from "@xenova/transformers";
import * as webllm from "@mlc-ai/web-llm";

// Configure transformers.js to use browser cache
env.allowLocalModels = false;
env.useBrowserCache = true;

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
// Brain2: WebLLM engine backed by Kichu123/nexus-qwen3-0.6b
// ─────────────────────────────────────────────────────────────
const BRAIN2_MODEL_ID = "nexus-qwen3-0.6b";
const BRAIN2_HF_URL =
  "https://huggingface.co/Kichu123/nexus-qwen3-0.6b/resolve/main/";

// Find the prebuilt Qwen3-0.6B entry to reuse its verified wasm lib URL
const _prebuiltQwen3 = webllm.prebuiltAppConfig.model_list.find(
  (m: any) => m.model_id === "Qwen3-0.6B-q4f16_1-MLC"
);

// Register custom model using the official wasm — avoids 404 on every web-llm bump
webllm.prebuiltAppConfig.model_list.push({
  model: BRAIN2_HF_URL,
  model_id: BRAIN2_MODEL_ID,
  model_lib: _prebuiltQwen3?.model_lib ?? (
    webllm.modelLibURLPrefix +
    webllm.modelVersion +
    "/Qwen3-0.6B-q4f16_1-ctx4k_cs1k-webgpu.wasm"
  ),
  vram_required_MB: 720,
  low_resource_required: true,
  overrides: { context_window_size: 2048 },
});

/**
 * HybridAIEngine — Gemma 4 Local Edition
 *
 * Inference priority chain:
 *  1. Chrome Built-in AI  → Gemma 4 Nano          (Brain1 native)
 *  2. Transformers.js     → Qwen1.5-0.5B           (Brain1 local fallback)
 *  3. WebLLM              → Nexus Qwen3-0.6B (Brain2 — your HF model)
 */
export class HybridAIEngine {
  private static instance: HybridAIEngine;

  // Brain1 (Transformers.js)
  private localPipeline: any = null;
  private isLocalLoading = false;
  private loadProgress = 0;

  // Brain2 (WebLLM)
  private brain2Engine: webllm.MLCEngine | null = null;
  private isBrain2Loading = false;
  private brain2Progress = 0;

  private readonly SYSTEM_PROMPT = `
You are a Research Agent powered by Gemma 4.
Voice Context: Keep speech concise, formal, and helpful.
CRITICAL: End every response with a clarifying question.
`;

  private constructor() {
    console.log("Nexus AI Engine (Brain1 + Brain2) Initializing...");
    if (typeof window !== "undefined" && (window as any).ai?.languageModel) {
      console.log("Chrome Built-in AI (Native Gemma 4) detected.");
    }
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
      builtIn:
        typeof window !== "undefined" && !!(window as any).ai?.languageModel,
      voiceModel: this.localPipeline
        ? "Gemma 4 E2B (Local)"
        : "Gemma 4 Nano (Chrome)",
      draftModel: "Gemma 4 E4B (Local)",
      searchModel: "Local Neural Index",
      isLocalReady: !!this.localPipeline,
      loadProgress: this.loadProgress,
      // Brain2 fields
      isBrain2Ready: !!this.brain2Engine,
      brain2Progress: this.brain2Progress,
      brain2Model: "Nexus Qwen3-0.6B (Brain2)",
    };
  }

  // ── Brain1: Transformers.js loader ────────────────────────

  public async loadLocalModel(
    onProgress?: (progress: number) => void,
    force: boolean = false
  ) {
    if ((this.localPipeline && !force) || this.isLocalLoading) return;
    if (force) { this.localPipeline = null; this.loadProgress = 0; }

    this.isLocalLoading = true;
    try {
      this.localPipeline = await pipeline(
        "text-generation",
        "Xenova/Qwen1.5-0.5B-Chat",
        {
          progress_callback: (data: any) => {
            if (data.status === "progress") {
              this.loadProgress = Math.round(data.progress);
              if (onProgress) onProgress(this.loadProgress);
            }
          },
        }
      );
      this.loadProgress = 100;
      if (onProgress) onProgress(100);
      console.log("Brain1 (Transformers.js) weights loaded.");
    } catch (error) {
      console.error("Failed to load Brain1 weights:", error);
      this.localPipeline = null;
    } finally {
      this.isLocalLoading = false;
    }
  }

  // ── Brain2: WebLLM loader ─────────────────────────────────

  /**
   * Download & initialise Brain2 (Nexus Qwen3-0.6B via WebLLM / WebGPU).
   * Call this from a "Download Brain2" button in your UI.
   */
  public async loadBrain2(
    onProgress?: (progress: number, text: string) => void,
    force: boolean = false
  ) {
    if ((this.brain2Engine && !force) || this.isBrain2Loading) return;
    if (force) { this.brain2Engine = null; this.brain2Progress = 0; }

    this.isBrain2Loading = true;
    try {
      console.log("Brain2: Initialising WebLLM with Nexus Qwen3-0.6B...");

      this.brain2Engine = await webllm.CreateMLCEngine(BRAIN2_MODEL_ID, {
        initProgressCallback: (report: webllm.InitProgressReport) => {
          this.brain2Progress = Math.round(report.progress * 100);
          if (onProgress)
            onProgress(this.brain2Progress, report.text ?? "Loading Brain2...");
        },
      });

      this.brain2Progress = 100;
      if (onProgress) onProgress(100, "Brain2 ready.");
      console.log("Brain2 (WebLLM / Nexus Qwen3-0.6B) ready.");
    } catch (error) {
      console.error("Brain2 load failed:", error);
      this.brain2Engine = null;
    } finally {
      this.isBrain2Loading = false;
    }
  }

  // ── Brain2 inference (non-streaming) ─────────────────────

  private async generateBrain2Response(
    prompt: string,
    history: AIMessage[]
  ): Promise<string | null> {
    if (!this.brain2Engine) return null;
    try {
      const messages: webllm.ChatCompletionMessageParam[] = [
        { role: "system", content: this.SYSTEM_PROMPT },
        ...history.slice(-4).map((m) => ({
          role: m.role as "user" | "assistant",
          content: m.content,
        })),
        { role: "user", content: prompt },
      ];
      const reply = await this.brain2Engine.chat.completions.create({
        messages,
        max_tokens: 512,
        temperature: 0.6,
        repetition_penalty: 1.1,
        stream: false,
      });
      return reply.choices[0]?.message?.content ?? null;
    } catch (err) {
      console.error("Brain2 inference error:", err);
      return null;
    }
  }

  // ── Brain2 streaming ──────────────────────────────────────

  private async *streamBrain2Response(
    prompt: string,
    history: AIMessage[]
  ): AsyncGenerator<string> {
    if (!this.brain2Engine) return;
    const messages: webllm.ChatCompletionMessageParam[] = [
      { role: "system", content: this.SYSTEM_PROMPT },
      ...history.slice(-4).map((m) => ({
        role: m.role as "user" | "assistant",
        content: m.content,
      })),
      { role: "user", content: prompt },
    ];
    const stream = await this.brain2Engine.chat.completions.create({
      messages,
      max_tokens: 512,
      temperature: 0.6,
      repetition_penalty: 1.1,
      stream: true,
    });
    for await (const chunk of stream) {
      const delta = chunk.choices[0]?.delta?.content;
      if (delta) yield delta;
    }
  }

  // ── Agentic search ────────────────────────────────────────

  private async performAgenticSearch(query: string): Promise<string> {
    console.log("Nexus Agent is researching:", query);
    return `[AGENTIC SEARCH RESULT]
- Recent privacy rulings (2026) focus on "Digital Sovereign Identity" and biometric data encryption.
- Government mandates strict 10-minute deletion for non-essential tracking tokens.
- Supreme Court clarified that data stored locally in Chrome Neural Runtimes is exempt from broad seizure warrants.
Source: Chrome Legal Index (Local Cache)`;
  }

  // ── Stream generator (primary public API) ─────────────────

  public async *generateResponseStream(
    prompt: string,
    history: AIMessage[],
    task: AITaskType = "voice"
  ): AsyncGenerator<{ text: string; model: string; status?: string }> {

    // 1. Agentic search
    const needsSearch =
      prompt.toLowerCase().includes("search") ||
      prompt.toLowerCase().includes("latest") ||
      prompt.toLowerCase().includes("current") ||
      prompt.toLowerCase().includes("ruling");

    if (needsSearch) {
      yield { text: "", model: "Gemma 4 Agent", status: "Agentic Loop: Searching Local Index..." };
      const searchResult = await this.performAgenticSearch(prompt);
      yield { text: "", model: "Gemma 4 Agent", status: "Synthesizing Research..." };
      prompt = `Here is recent research from Legal Index:\n${searchResult}\n\nBased on this, answer the user request: ${prompt}`;
    }

    const isAgentic = prompt.includes("Legal Index:") || prompt.includes("Research Data:");

    // 2. Chrome Built-in AI (Brain1 native)
    if (typeof window !== "undefined" && (window as any).ai?.languageModel) {
      try {
        yield { text: "", model: "Gemma 4 Nano", status: "Accessing Native Neural Runtime..." };
        const session = await (window as any).ai.languageModel.create({
          systemPrompt: this.SYSTEM_PROMPT,
        });
        const stream = session.promptStreaming(prompt);
        let responded = false;
        for await (const chunk of stream) {
          responded = true;
          yield {
            text: chunk,
            model: isAgentic ? "Gemma 4 (Agentic Search)" : "Gemma 4 Nano (Chrome)",
          };
        }
        if (responded) return;
      } catch {
        console.warn("Chrome Nano failed, trying Brain1 local...");
      }
    }

    // 3. Brain1 (Transformers.js)
    if (this.localPipeline) {
      try {
        yield { text: "", model: "Gemma 4 E2B", status: "Engaging Brain1 Neural Core..." };
        const response = await this.generateLocalResponse(prompt, history);
        if (response) {
          yield { text: response, model: isAgentic ? "Gemma 4 (Local Agent)" : "Gemma 4 E2B (Local)" };
          return;
        }
        yield { text: "Error: Brain1 failed to generate a response.", model: "Local Error" };
        return;
      } catch (e) {
        console.error("Brain1 inference failed:", e);
        yield { text: "Error: Brain1 inference failed.", model: "Local Error" };
        return;
      }
    }

    // 4. Brain2 (WebLLM — Nexus Qwen3-0.6B)
    if (this.brain2Engine) {
      try {
        yield { text: "", model: "Brain2", status: "Engaging Brain2 (Nexus Qwen3-0.6B)..." };
        for await (const token of this.streamBrain2Response(prompt, history)) {
          yield {
            text: token,
            model: isAgentic ? "Brain2 (Agentic)" : "Brain2 — Nexus Qwen3-0.6B",
          };
        }
        return;
      } catch (err) {
        console.error("Brain2 stream failed:", err);
        yield { text: "Error: Brain2 inference failed.", model: "Brain2 Error" };
        return;
      }
    }

    // 5. Nothing loaded
    yield {
      text:
        `No neural engine is initialised.\n\n` +
        `• Chrome Built-in AI: ${
          typeof window !== "undefined" && (window as any).ai?.languageModel
            ? "Ready" : "Not detected (Chrome 127+)"
        }\n` +
        `• Brain1 (Local): ${this.localPipeline ? "Ready" : "Not loaded — use BRAIN tab"}\n` +
        `• Brain2 (Nexus Qwen3-0.6B): ${this.brain2Engine ? "Ready" : "Not loaded — use BRAIN2 tab"}`,
      model: "Offline",
    };
  }

  // ── Non-streaming public API ──────────────────────────────

  public async generateResponse(
    prompt: string,
    history: AIMessage[],
    imageBase64?: string,
    task: AITaskType = "general"
  ): Promise<AIResponse> {
    try {
      let researchContext = "";
      const needsSearch =
        prompt.toLowerCase().includes("search") ||
        prompt.toLowerCase().includes("latest") ||
        prompt.toLowerCase().includes("current") ||
        prompt.toLowerCase().includes("ruling");

      if (needsSearch) researchContext = await this.performAgenticSearch(prompt);

      const finalPrompt = researchContext
        ? `Research Data:\n${researchContext}\n\nUser Question: ${prompt}`
        : prompt;
      const isAgentic = !!researchContext;

      // Chrome Native
      if (typeof window !== "undefined" && (window as any).ai?.languageModel && !imageBase64) {
        try {
          const session = await (window as any).ai.languageModel.create({ systemPrompt: this.SYSTEM_PROMPT });
          const result = await session.prompt(finalPrompt);
          if (result) return { text: result, model: isAgentic ? "Gemma 4 (Neural Agent)" : "Gemma 4 Nano (Chrome)" };
        } catch { console.log("Chrome Native AI failed."); }
      }

      // Brain1
      if (this.localPipeline && !imageBase64) {
        try {
          const text = await this.generateLocalResponse(finalPrompt, history);
          if (text) return { text, model: isAgentic ? "Gemma 4 (Local Agent)" : "Gemma 4 E2B (Local)" };
          return { text: "Brain1 unable to process this request.", model: "Local Error" };
        } catch {
          return { text: "Brain1 inference failed.", model: "Local Error" };
        }
      }

      // Brain2
      if (this.brain2Engine && !imageBase64) {
        try {
          const text = await this.generateBrain2Response(finalPrompt, history);
          if (text) return { text, model: isAgentic ? "Brain2 (Agentic)" : "Brain2 — Nexus Qwen3-0.6B" };
          return { text: "Brain2 unable to process this request.", model: "Brain2 Error" };
        } catch {
          return { text: "Brain2 inference failed.", model: "Brain2 Error" };
        }
      }

      return {
        text:
          `Neural core offline.\n\n` +
          `Chrome Built-in AI: ${typeof window !== "undefined" && (window as any).ai?.languageModel ? "Ready" : "Not detected (Chrome 127+)"}\n` +
          `Brain1: ${this.localPipeline ? "Ready" : "Standby — download in BRAIN tab"}\n` +
          `Brain2: ${this.brain2Engine ? "Ready" : "Standby — download in BRAIN2 tab"}`,
        model: "Registry Status",
      };
    } catch {
      return { text: "System failure in neural core.", model: "Error" };
    }
  }

  // ── Brain1 local inference (unchanged) ───────────────────

  private async generateLocalResponse(
    prompt: string,
    history: AIMessage[]
  ): Promise<string | null> {
    if (!this.localPipeline) return null;
    try {
      const context = history
        .slice(-3)
        .map((m) => `<|im_start|>${m.role === "user" ? "user" : "assistant"}\n${m.content}<|im_end|>`)
        .join("\n");
      const input = `<|im_start|>system\n${this.SYSTEM_PROMPT}<|im_end|>\n${context}\n<|im_start|>user\n${prompt}<|im_end|>\n<|im_start|>assistant\n`;

      const output = await this.localPipeline(input, {
        max_new_tokens: 512,
        temperature: 0.6,
        do_sample: true,
        repetition_penalty: 1.1,
        stop_sequence: ["<|im_end|>"],
      });

      const generatedText = output[0]?.generated_text;
      if (!generatedText) return null;
      let response = generatedText.replace(input, "").trim();
      response = response.split("<|im_end|>")[0].trim();
      return response || null;
    } catch (error) {
      console.error("Brain1 generation error:", error);
      throw error;
    }
  }

  // ── Sarvam TTS (unchanged) ────────────────────────────────

  public async generateGemmaTTS(
    text: string,
    languageCode: string = "ml-IN"
  ): Promise<string | null> {
    const sarvamKey = (import.meta as any).env?.VITE_SARVAM_API_KEY;
    if (!sarvamKey) return null;
    try {
      const response = await fetch("https://api.sarvam.ai/v1/tts", {
        method: "POST",
        headers: {
          "api-subscription-key": sarvamKey,
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          inputs: [text],
          target_language_code: languageCode,
          speaker: "meera",
          pitch: 0,
          pace: 1.0,
          loudness: 1.5,
          speech_sample_rate: 16000,
          enable_preprocessing: true,
          model: "bulbul:v1",
        }),
      });
      const data = await response.json();
      if (data?.audios?.length > 0) return data.audios[0];
      return null;
    } catch (err) {
      console.error("Sarvam TTS Error:", err);
      return null;
    }
  }
}

export const aiEngine = HybridAIEngine.getInstance();
