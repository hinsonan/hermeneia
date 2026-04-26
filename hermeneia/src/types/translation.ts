// Translation types for translation UI and queue scheduling

export type TranslationPhase =
  | "waiting_resources"
  | "downloading_model"
  | "loading_model"
  | "translating"
  | "completed";

export interface TranslationProgress {
  job_id?: string;
  phase: TranslationPhase;
  current: number | null;
  total: number | null;
  message: string;
}

// Result from translate_text_file command
export interface TextTranslationResult {
  translated_text: string;
  original_text: string;
  is_subtitle: boolean;
  source_language: string;
  target_language: string;
  model_used: string;
  inference_time: number;
  segments_translated: number;
}

// Language option for dropdowns
export interface LanguageOption {
  code: string;
  name: string;
}

export type TranslationJobStatus =
  | "queued"
  | "waiting_resources"
  | "downloading_model"
  | "loading_model"
  | "running"
  | "cancelling"
  | "completed"
  | "failed"
  | "cancelled";

export type TranslationFailureType =
  | "oom"
  | "model_load"
  | "transient"
  | "unsupported"
  | "cancelled"
  | "unknown";

export type TranslationStrategy = "auto" | "fast_only" | "universal";

export type TranslationEngineTier = "fast" | "universal";

export interface TranslationEngineMetadata {
  tier: TranslationEngineTier;
  label: string;
  message: string;
  modelId: string;
  modelName: string;
}

export interface TranslationJobSettings {
  sourceLang: string;
  targetLang: string;
  strategy: TranslationStrategy;
}

export interface TranslationQueueJob {
  id: string;
  batchId: string;
  createdAt: number;
  filePath: string;
  fileName: string;
  status: TranslationJobStatus;
  settings: TranslationJobSettings;
  engine: TranslationEngineMetadata | null;
  progress: TranslationProgress | null;
  downloadProgress: import("./models").DownloadProgress | null;
  result: TextTranslationResult | null;
  error: string | null;
  retryCount: number;
  lastFailureType: TranslationFailureType | null;
}

export interface InferenceRuntimeLimits {
  max_inference_concurrency: number;
}

// Supported translation languages
export const MADLAD_LANGUAGES: LanguageOption[] = [
  { code: "en", name: "English" },
  { code: "es", name: "Spanish" },
  { code: "fr", name: "French" },
  { code: "de", name: "German" },
  { code: "pt", name: "Portuguese" },
  { code: "it", name: "Italian" },
  { code: "ru", name: "Russian" },
  { code: "zh", name: "Chinese" },
  { code: "ja", name: "Japanese" },
  { code: "ko", name: "Korean" },
  { code: "ar", name: "Arabic" },
  { code: "nl", name: "Dutch" },
  { code: "pl", name: "Polish" },
  { code: "tr", name: "Turkish" },
  { code: "vi", name: "Vietnamese" },
  { code: "th", name: "Thai" },
  { code: "id", name: "Indonesian" },
  { code: "hi", name: "Hindi" },
  { code: "he", name: "Hebrew" },
  { code: "el", name: "Greek" },
  { code: "sv", name: "Swedish" },
  { code: "da", name: "Danish" },
  { code: "no", name: "Norwegian" },
  { code: "fi", name: "Finnish" },
  { code: "cs", name: "Czech" },
  { code: "uk", name: "Ukrainian" },
  { code: "ro", name: "Romanian" },
  { code: "hu", name: "Hungarian" },
  { code: "fa", name: "Persian" },
  { code: "bn", name: "Bengali" },
  { code: "ur", name: "Urdu" },
  { code: "sw", name: "Swahili" },
];

// Get language name from code
export function getLanguageName(code: string): string {
  const lang = MADLAD_LANGUAGES.find((l) => l.code === code);
  return lang ? lang.name : code.toUpperCase();
}
