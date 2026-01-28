// Translation types for the Translation page

// Progress reporting types
export type TranslationPhase = 'loading_model' | 'translating';

export interface TranslationProgress {
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

// Marian-supported languages (subset that has Marian models)
// All pairs involve English as either source or target
export const MADLAD_LANGUAGES: LanguageOption[] = [
  { code: 'en', name: 'English' },
  { code: 'es', name: 'Spanish' },
  { code: 'fr', name: 'French' },
  { code: 'de', name: 'German' },
  { code: 'pt', name: 'Portuguese' },
  { code: 'it', name: 'Italian' },
  { code: 'ru', name: 'Russian' },
  { code: 'zh', name: 'Chinese' },
  { code: 'ja', name: 'Japanese' },
  { code: 'ko', name: 'Korean' },
  { code: 'ar', name: 'Arabic' },
  { code: 'nl', name: 'Dutch' },
  { code: 'pl', name: 'Polish' },
  { code: 'tr', name: 'Turkish' },
  { code: 'vi', name: 'Vietnamese' },
  { code: 'th', name: 'Thai' },
  { code: 'id', name: 'Indonesian' },
  { code: 'hi', name: 'Hindi' },
  { code: 'he', name: 'Hebrew' },
  { code: 'el', name: 'Greek' },
  { code: 'sv', name: 'Swedish' },
  { code: 'da', name: 'Danish' },
  { code: 'no', name: 'Norwegian' },
  { code: 'fi', name: 'Finnish' },
  { code: 'cs', name: 'Czech' },
  { code: 'uk', name: 'Ukrainian' },
  { code: 'ro', name: 'Romanian' },
  { code: 'hu', name: 'Hungarian' },
  { code: 'fa', name: 'Persian' },
  { code: 'bn', name: 'Bengali' },
  { code: 'ur', name: 'Urdu' },
  { code: 'sw', name: 'Swahili' },
];

// Get language name from code
export function getLanguageName(code: string): string {
  const lang = MADLAD_LANGUAGES.find(l => l.code === code);
  return lang ? lang.name : code.toUpperCase();
}
