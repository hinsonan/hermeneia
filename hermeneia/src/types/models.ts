// Types for the Model Library page

export interface ModelInfo {
  model_id: string;
  display_name: string;
  category: "whisper" | "marian" | "madlad";
  size_mb: number;
  is_cached: boolean;
  description: string;
  source_lang: string | null;
  target_lang: string | null;
}

export interface DownloadProgress {
  model_id: string;
  model_name: string;
  file_name: string;
  file_index: number;
  total_files: number;
  bytes_downloaded: number;
  bytes_total: number | null;
  phase: "downloading" | "complete" | "cancelled";
}
