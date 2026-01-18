export type WhisperModel = 'tiny' | 'tiny.en' | 'base' | 'base.en' | 'small' | 'small.en' | 'medium' | 'medium.en' | 'large' | 'large-v2' | 'large-v3';
export type TranscriptionTask = 'transcribe' | 'translate';

// Progress reporting types
export type TranscriptionPhase = 'loading_model' | 'transcribing';

export interface TranscriptionProgress {
  phase: TranscriptionPhase;
  current: number | null;
  total: number | null;
  message: string;
}

export interface TranscriptSegment {
  id: number;
  start: number | null;
  end: number | null;
  text: string;
}

export interface TranscriptResult {
  segments: TranscriptSegment[];
  text: string;
  language: string | null;
  duration: number;
  model: WhisperModel;
  inference_time: number;
}

export interface ModelOption {
  value: WhisperModel;
  label: string;
  description: string;
}

export interface LanguageOption {
  value: string | null;
  label: string;
}

// System capability detection types
export type GpuDeviceType = 'NvidiaCuda' | 'AmdRocm' | 'AppleMetal' | 'None';

export interface GpuInfo {
  device_type: GpuDeviceType;
  vram_total_gb: number | null;
  vram_available_gb: number | null;
  compute_capability: [number, number] | null;
}

export interface SystemCapabilities {
  total_ram_gb: number;
  available_ram_gb: number;
  gpu_info: GpuInfo | null;
}

export interface ModelValidation {
  status: 'ok' | 'warning' | 'error';
  messages: string[];
  recommended_model: string | null;
}
