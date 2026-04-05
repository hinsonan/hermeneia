export type WhisperModel = 'tiny' | 'tiny.en' | 'base' | 'base.en' | 'small' | 'small.en' | 'medium' | 'medium.en' | 'large' | 'large-v2' | 'large-v3';
export type TranscriptionTask = 'transcribe' | 'translate';

// Progress reporting types
export type TranscriptionPhase =
  | 'decoding_audio'
  | 'preparing_audio'
  | 'loading_model'
  | 'transcribing'
  | 'completed';

export interface TranscriptionProgress {
  job_id: string;
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

// Annotation types (merged into transcription page flow)
export type SpeakerModelKey = 'english' | 'multilingual';
export type SpeakerDevice = 'cpu' | 'cuda' | 'coreml';
export type AnnotationPhase =
  | 'starting'
  | 'decoding_audio'
  | 'preparing_audio'
  | 'loading_speaker_model'
  | 'ensuring_speaker_models'
  | 'initializing_speaker_runtime'
  | 'diarizing'
  | 'loading_transcription_model'
  | 'transcribing'
  | 'merging'
  | 'completed';

export interface AnnotationProgress {
  job_id: string;
  phase: AnnotationPhase;
  current: number | null;
  total: number | null;
  message: string;
  indeterminate: boolean;
}

export interface AnnotatedSegment {
  index: number;
  start: number;
  end: number;
  speaker: number;
  speaker_name: string;
  text: string;
}

export interface AnnotatedResult {
  segments: AnnotatedSegment[];
  speaker_names: Record<string, string>;
  num_speakers: number;
  language: string | null;
  audio_duration: number;
  diarization_inference_time: number;
  transcription_inference_time: number;
  total_inference_time: number;
  whisper_model: string;
  speaker_model: string;
  speaker_device: string;
}

export interface SpeakerModelRequirement {
  key: SpeakerModelKey;
  display_name: string;
  approx_size_mb: number;
  is_cached: boolean;
  segmentation_model_id: string;
  segmentation_file: string;
  embedding_model_id: string;
  embedding_file: string;
}
